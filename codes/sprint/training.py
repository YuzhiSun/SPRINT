"""
Training engine and evaluation utilities for SPRINT models.

Provides:
- PCCLoss : Pearson-correlation-based loss for multi-target regression
- train_engine : Full training loop with checkpointing and history logging
- evaluate_comprehensive : Multi-metric evaluation (PCC, RMSE, MAE, R²)
- Utility functions for loading histories, models, and plotting
"""

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ===========================================================================
# Metrics
# ===========================================================================

def calculate_pcc(preds, targets):
    """Mean per-protein Pearson correlation coefficient."""
    pccs = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(preds.shape[1]):
            p = preds[:, i]
            t = targets[:, i]
            if np.std(p) == 0 or np.std(t) == 0:
                pccs.append(0.0)
            else:
                res = pearsonr(p, t)[0]
                pccs.append(res if not np.isnan(res) else 0.0)
    return float(np.mean(pccs))


# ===========================================================================
# Loss
# ===========================================================================

class PCCLoss(nn.Module):
    """1 − Pearson correlation loss for multi-target regression."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds, targets):
        if preds.dim() == 3:
            preds = preds.squeeze(1)
        if targets.dim() == 3:
            targets = targets.squeeze(1)
        vx = preds - torch.mean(preds, dim=0, keepdim=True)
        vy = targets - torch.mean(targets, dim=0, keepdim=True)
        cost = torch.sum(vx * vy, dim=0)
        denom = torch.sqrt(
            torch.sum(vx ** 2, dim=0) * torch.sum(vy ** 2, dim=0)
        ) + self.eps
        return 1.0 - torch.mean(cost / denom)


# ===========================================================================
# Model instantiation helper
# ===========================================================================

def instantiate_model(model_cls, num_proteins, num_genes, device):
    """Instantiate a model class with try/except for num_genes kwarg."""
    try:
        return model_cls(num_proteins=num_proteins, num_genes=num_genes).to(device)
    except TypeError:
        return model_cls(num_proteins=num_proteins).to(device)


# ===========================================================================
# Training loop
# ===========================================================================

def train_engine(
    model_cls,
    name,
    train_loader,
    val_loader,
    num_proteins,
    device,
    epochs=30,
    lr=1e-4,
    model_kwargs=None,
    save_root="models",
):
    """Full training loop with checkpointing.

    Parameters
    ----------
    model_cls : nn.Module class
        Model class to instantiate.
    name : str
        Name for the model run (used for save directory).
    train_loader, val_loader : DataLoader
    num_proteins : int
        Number of output targets.
    device : torch.device
    epochs : int
    lr : float
        Learning rate for Adam optimizer.
    model_kwargs : dict, optional
        Extra kwargs passed to model constructor.
    save_root : str
        Root directory for checkpoints and history.

    Returns
    -------
    history : dict
        Training history with keys: train_loss, val_loss, val_pcc, val_rmse, val_r2.
    """
    model_kwargs = model_kwargs or {}
    save_dir = os.path.join(save_root, name)
    os.makedirs(save_dir, exist_ok=True)

    ckpt_path = os.path.join(save_dir, "ckpt.pth")
    best_path = os.path.join(save_dir, "best_model.pth")
    history_path = os.path.join(save_dir, "history.json")

    model = model_cls(num_proteins=num_proteins, **model_kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_mse = nn.MSELoss()
    criterion_pcc = PCCLoss()

    history = {
        "train_loss": [], "val_loss": [], "val_pcc": [], "val_rmse": [], "val_r2": [],
    }
    start_epoch = 0
    best_pcc = -1.0

    # Resume from checkpoint if available
    if os.path.exists(ckpt_path):
        try:
            print(f"Found checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get("model", ckpt.get("model_state_dict", ckpt)))
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            history = ckpt.get("history", history)
            start_epoch = ckpt.get("epoch", -1) + 1
            best_pcc = ckpt.get("best_pcc", best_pcc)
            if start_epoch >= epochs:
                print(f"Already finished {start_epoch} epochs. Skipping.")
                return history
            print(f"Resuming from Epoch {start_epoch + 1} to {epochs}...")
        except Exception as exc:
            print(f"Resume failed ({exc}). Restarting training.")

    # Fallback: load history from JSON
    if os.path.exists(history_path) and not history["train_loss"]:
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0

        for img, rna, label in tqdm(
            train_loader, desc=f"[{name}] Ep {epoch + 1}/{epochs}", leave=False
        ):
            img, rna, label = img.to(device), rna.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(img, rna)
            if out.dim() == 3:
                out = out.squeeze(1)
            loss = 0.5 * criterion_mse(out, label) + 0.5 * criterion_pcc(out, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        preds_list, labels_list = [], []
        with torch.no_grad():
            for img, rna, label in val_loader:
                img, rna, label = img.to(device), rna.to(device), label.to(device)
                out = model(img, rna)
                if out.dim() == 3:
                    out = out.squeeze(1)
                loss = 0.5 * criterion_mse(out, label) + 0.5 * criterion_pcc(out, label)
                val_loss += loss.item()
                preds_list.append(out.cpu().numpy())
                labels_list.append(label.cpu().numpy())

        val_loss /= max(len(val_loader), 1)
        preds = np.vstack(preds_list)
        labels = np.vstack(labels_list)
        val_pcc = calculate_pcc(preds, labels)
        val_rmse = float(np.sqrt(mean_squared_error(labels, preds)))
        try:
            val_r2 = float(r2_score(labels, preds))
        except Exception:
            val_r2 = -999.0

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_pcc"].append(float(val_pcc))
        history["val_rmse"].append(float(val_rmse))
        history["val_r2"].append(float(val_r2))

        print(
            f"   Ep {epoch + 1}: TrainLoss={train_loss:.4f}, "
            f"ValLoss={val_loss:.4f} | PCC={val_pcc:.4f}, RMSE={val_rmse:.4f}"
        )

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

        if val_pcc > best_pcc:
            best_pcc = val_pcc
            torch.save(model.state_dict(), best_path)

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "best_pcc": best_pcc,
            },
            ckpt_path,
        )

    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history


# ===========================================================================
# Evaluation
# ===========================================================================

def load_model_weights(model, path, device):
    """Load model weights from a checkpoint file."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                model.load_state_dict(checkpoint[key])
                return model
    model.load_state_dict(checkpoint)
    return model


def evaluate_comprehensive(model, loader, device):
    """Run evaluation on a DataLoader and return PCC, RMSE, MAE, R²."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for img, rna, label in loader:
            img, rna, label = img.to(device), rna.to(device), label.to(device)
            out = model(img, rna)
            if out.dim() == 3:
                out = out.squeeze(1)
            all_preds.append(out.cpu().numpy())
            all_targets.append(label.cpu().numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    return {
        "PCC": calculate_pcc(preds, targets),
        "RMSE": float(np.sqrt(mean_squared_error(targets, preds))),
        "MAE": float(mean_absolute_error(targets, preds)),
        "R2": float(r2_score(targets, preds)),
    }


def calculate_pcc_single(y_true, y_pred):
    """Single-target Pearson correlation."""
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def evaluate_model_per_protein(
    model_cls, model_name, loader, device, protein_names,
    model_kwargs=None, save_root="models",
):
    """Per-protein evaluation: returns a DataFrame with PCC/RMSE/R² per protein."""
    model_kwargs = model_kwargs or {}
    path = os.path.join(save_root, model_name, "best_model.pth")
    if not os.path.exists(path):
        print(f"{model_name}: weights not found at {path}")
        return None

    model = instantiate_model(model_cls, len(protein_names), 0, device)
    try:
        load_model_weights(model, path, device)
    except Exception as exc:
        print(f"Error loading {model_name}: {exc}")
        return None

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for img, rna, label in loader:
            img, rna, label = img.to(device), rna.to(device), label.to(device)
            out = model(img, rna)
            if out.dim() == 3:
                out = out.squeeze(1)
            all_preds.append(out.cpu().numpy())
            all_targets.append(label.cpu().numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    results = []
    for i, name in enumerate(protein_names):
        yt, yp = targets[:, i], preds[:, i]
        mse = mean_squared_error(yt, yp)
        results.append({
            "Model": model_name,
            "Protein": name,
            "PCC": calculate_pcc_single(yt, yp),
            "RMSE": float(np.sqrt(mse)),
            "R2": float(r2_score(yt, yp)),
        })
    return pd.DataFrame(results)


def evaluate_models(
    model_specs, loader, num_proteins, device, model_kwargs=None, save_root="models",
):
    """Evaluate a list of (model_cls, name) tuples and return a metrics DataFrame."""
    metrics_data = []
    model_kwargs = model_kwargs or {}
    for model_cls, model_name in model_specs:
        path = os.path.join(save_root, model_name, "best_model.pth")
        if not os.path.exists(path):
            print(f"Model {model_name} not found, skipping.")
            continue
        print(f"Evaluating: {model_name}...")
        eval_model = instantiate_model(model_cls, num_proteins, 0, device)
        try:
            load_model_weights(eval_model, path, device)
        except Exception as exc:
            print(f"Error loading {model_name}: {exc}")
            continue
        metrics = evaluate_comprehensive(eval_model, loader, device)
        metrics["Model"] = model_name
        metrics_data.append(metrics)
    return pd.DataFrame(metrics_data)


def evaluate_per_protein_models(
    model_specs, loader, device, protein_names, model_kwargs=None, save_root="models",
):
    """Per-protein evaluation across all models in model_specs."""
    all_results = []
    for model_cls, model_name in model_specs:
        result = evaluate_model_per_protein(
            model_cls, model_name, loader, device, protein_names,
            model_kwargs=model_kwargs, save_root=save_root,
        )
        if result is not None:
            all_results.append(result)
    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


def plot_metrics(df_metrics, save_path="models/comprehensive_comparison.png"):
    """Plot comprehensive model comparison: PCC, RMSE, R², and efficiency frontier."""
    if df_metrics.empty:
        print("No models evaluated for plotting.")
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    _, axes = plt.subplots(2, 2, figsize=(16, 12))
    palette = sns.color_palette("viridis", n_colors=len(df_metrics))

    sns.barplot(data=df_metrics, x="Model", y="PCC", ax=axes[0, 0], palette=palette)
    axes[0, 0].set_title("Pearson Correlation (Higher is Better)", fontsize=14, fontweight="bold")
    axes[0, 0].set_ylim(0, 1.0)

    sns.barplot(data=df_metrics, x="Model", y="RMSE", ax=axes[0, 1], palette="magma")
    axes[0, 1].set_title("RMSE (Lower is Better)", fontsize=14, fontweight="bold")

    sns.barplot(data=df_metrics, x="Model", y="R2", ax=axes[1, 0], palette="rocket")
    axes[1, 0].set_title("R² Score", fontsize=14, fontweight="bold")

    sns.scatterplot(
        data=df_metrics, x="RMSE", y="PCC", hue="Model",
        style="Model", s=200, ax=axes[1, 1], palette=palette,
    )
    axes[1, 1].set_title("Efficiency Frontier: RMSE vs PCC", fontsize=14, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Comparison plot saved to {save_path}")


# ===========================================================================
# History loading & comparison
# ===========================================================================

def load_histories(model_specs, save_root="models"):
    """Load training history JSONs for a list of (model_cls, name) tuples."""
    histories = {}
    for _, name in model_specs:
        path = os.path.join(save_root, name, "history.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                histories[name] = json.load(f)
    return histories


def plot_comparison(histories, save_path="models/comparison_result.png"):
    """Plot validation PCC curves for multiple training histories."""
    if not histories:
        print("No history data found for comparison.")
        return

    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 8))
    for name, hist in histories.items():
        if hist.get("val_pcc"):
            plt.plot(hist["val_pcc"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation PCC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Comparison plot saved to {save_path}")


# ===========================================================================
# MSI-specific training (warmup-cosine scheduler, differential LR, z-score)
# ===========================================================================

class StablePCCLoss(nn.Module):
    """Numerically stable 1 − PCC loss for MSI metabolomics."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, preds, targets):
        vx = preds - preds.mean(dim=0, keepdim=True)
        vy = targets - targets.mean(dim=0, keepdim=True)
        var_prod = torch.sum(vx * vx, dim=0) * torch.sum(vy * vy, dim=0)
        denom = torch.sqrt(torch.clamp(var_prod, min=self.eps))
        return 1.0 - torch.mean(torch.sum(vx * vy, dim=0) / denom)


def pcc_mean(preds, targets):
    """Mean per-target PCC for numpy arrays."""
    from scipy.stats import pearsonr
    vals = []
    for i in range(preds.shape[1]):
        p, t = preds[:, i], targets[:, i]
        if np.std(p) == 0 or np.std(t) == 0:
            vals.append(0.0)
        else:
            r = pearsonr(p, t)[0]
            vals.append(0.0 if np.isnan(r) else float(r))
    return float(np.mean(vals))


def evaluate_msi(model, loader, mean, std, device):
    """MSI evaluation with z-score denormalization."""
    model.eval()
    preds_z, targets_z = [], []
    with torch.no_grad():
        for img, rna, y in loader:
            out = model(img.to(device), rna.to(device))
            preds_z.append(out.cpu().numpy())
            targets_z.append(y.numpy())
    preds_z = np.vstack(preds_z)
    targets_z = np.vstack(targets_z)
    preds = preds_z * std + mean
    targets = targets_z * std + mean
    return {
        "pcc_z": pcc_mean(preds_z, targets_z),
        "pcc": pcc_mean(preds, targets),
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
        "r2": float(r2_score(targets, preds)),
        "preds": preds,
        "targets": targets,
    }


def make_optimizer(model, lr, image_lr_factor, weight_decay):
    """Create optimizer with differential LR for image tower."""
    image_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("img_tower"):
            image_params.append(param)
        else:
            other_params.append(param)
    groups = [{"params": other_params, "lr": lr, "weight_decay": weight_decay}]
    if image_params:
        groups.append({"params": image_params, "lr": lr * image_lr_factor, "weight_decay": weight_decay})
    return optim.AdamW(groups)


def make_warmup_cosine_scheduler(optimizer, total_epochs, warmup_epochs=3, min_lr_factor=0.05):
    """Linear warmup + cosine decay scheduler."""
    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = max(int(warmup_epochs), 0)
    min_lr_factor = float(min_lr_factor)

    def lr_lambda(epoch_idx):
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(warmup_epochs)
        if total_epochs <= warmup_epochs:
            return 1.0
        progress = (epoch_idx - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_lr_dict(optimizer):
    """Return current learning rates for each param group."""
    return {f"lr_group_{i}": float(group["lr"]) for i, group in enumerate(optimizer.param_groups)}


def train_one(args, scheme, train_loader, val_loader, num_genes,
              num_targets, target_mean, target_std, out_dir):
    """MSI-specific training loop with warmup-cosine scheduling."""
    from .models import SPRINT_MSI

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SPRINT_MSI(scheme, num_genes, num_targets, freeze_image=args.freeze_image).to(device)
    optimizer = make_optimizer(model, args.lr, args.image_lr_factor, args.weight_decay)
    scheduler = make_warmup_cosine_scheduler(
        optimizer, total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs, min_lr_factor=args.min_lr_factor,
    )
    mse = nn.MSELoss()
    pcc = StablePCCLoss()
    history = {
        "train_loss": [], "val_pcc": [], "val_rmse": [], "val_r2": [],
        "lr_group_0": [], "lr_group_1": [],
    }
    best_pcc = -1.0
    best_path = Path(out_dir) / "best_model.pth" if not isinstance(out_dir, Path) else out_dir / "best_model.pth"
    ckpt_path = Path(out_dir) / "checkpoint.pth" if not isinstance(out_dir, Path) else out_dir / "checkpoint.pth"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0

    if getattr(args, "resume", False) and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        history = ckpt.get("history", history)
        best_pcc = float(ckpt.get("best_pcc", max(history.get("val_pcc", [-1.0]))))
        start_epoch = int(ckpt.get("epoch", -1)) + 1

    if start_epoch >= args.epochs:
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        final = evaluate_msi(model, val_loader, target_mean, target_std, device)
        return model, final, history

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = 0.0
        for step, (img, rna, y) in enumerate(train_loader, start=1):
            img, rna, y = img.to(device), rna.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(img, rna)
            loss = args.mse_weight * mse(out, y) + args.pcc_weight * pcc(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += loss.item()

        metrics = evaluate_msi(model, val_loader, target_mean, target_std, device)
        history["train_loss"].append(total / len(train_loader))
        history["val_pcc"].append(metrics["pcc"])
        history["val_rmse"].append(metrics["rmse"])
        history["val_r2"].append(metrics["r2"])
        current_lrs = get_lr_dict(optimizer)
        for key, value in current_lrs.items():
            history.setdefault(key, []).append(value)

        state = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_pcc": max(best_pcc, metrics["pcc"]),
            "history": history, "target_mean": target_mean,
            "target_std": target_std, "args": vars(args), "scheme": scheme,
        }
        torch.save(state, ckpt_path)
        if metrics["pcc"] > best_pcc:
            best_pcc = metrics["pcc"]
            torch.save(state, best_path)
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        scheduler.step()

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final = evaluate_msi(model, val_loader, target_mean, target_std, device)
    return model, final, history


def save_per_metabolite(preds, targets, names, out_path):
    """Save per-metabolite PCC, RMSE, R² metrics to CSV."""
    rows = []
    for i, name in enumerate(names):
        p, t = preds[:, i], targets[:, i]
        if np.std(p) == 0 or np.std(t) == 0:
            pcc_val = 0.0
        else:
            r = pearsonr(p, t)[0]
            pcc_val = 0.0 if np.isnan(r) else float(r)
        rows.append({
            "Metabolite": name,
            "PCC": pcc_val,
            "RMSE": float(np.sqrt(mean_squared_error(t, p))),
            "R2": float(r2_score(t, p)),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
