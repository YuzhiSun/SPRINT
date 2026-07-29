import gc
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BrainMultimodalDataset(Dataset):
    def __init__(self, h5ad_path, transform=None):
        print(f"Loading data from: {h5ad_path} ...")
        self.adata = sc.read_h5ad(h5ad_path)
        
        # 1. 图像数据 (Input A)
        if 'spatial_img_crops' in self.adata.obsm:
            self.images = self.adata.obsm['spatial_img_crops']
        else:
            print("Key 'spatial_img_crops' not found. Checking alternatives...")
            # Fallback check
            print(f"   Available Keys: {self.adata.obsm.keys()}")
            raise KeyError("spatial_img_crops not found in obsm")

        print(f"Raw Images Type: {type(self.images)}")
        if hasattr(self.images, 'shape'):
            print(f"Raw Images Shape: {self.images.shape}")
            
        # 检查并转换稀疏矩阵
        if scipy.sparse.issparse(self.images):
            print("Warning: Images are sparse matrix in __init__.")
            self.images = self.images.toarray() # Convert once to avoid repeated lag
        
        # 2. RNA 数据 (Input B)
        print(f"Using ALL Genes (No HVG filter).")
        X_data = self.adata.X
        self.gene_names = self.adata.var_names.tolist()
            
        if scipy.sparse.issparse(X_data):
            self.rna_data = X_data.toarray().astype(np.float32)
        else:
            self.rna_data = X_data.astype(np.float32)
            
        print(f"RNA Input Shape: {self.rna_data.shape}")

        # 3. 蛋白数据 (Target)
        self.protein_data = self.adata.obsm['protein_expression_log']
        if hasattr(self.protein_data, "values"):
            self.protein_data = self.protein_data.values
        if scipy.sparse.issparse(self.protein_data):
            self.protein_data = self.protein_data.toarray()
        self.protein_data = self.protein_data.astype(np.float32)
        
        # 4. 元数据
        if 'protein_names' in self.adata.uns:
            self.protein_names = self.adata.uns['protein_names']
        else:
            self.protein_names = [f"Prot_{i}" for i in range(self.protein_data.shape[1])]
            
        self.coords = self.adata.obsm['spatial']
        self.transform = transform

    def __len__(self):
        return len(self.rna_data)

    def __getitem__(self, idx):
        # A. 图像
        img = self.images[idx]
        
        # 1. 确保是 Numpy Array
        if not isinstance(img, np.ndarray):
             if torch.is_tensor(img): img = img.numpy()
             else: img = np.array(img)
            
        # 2. 维度检查与 Reshape
        # Case 1: (H, W) -> Grayscale -> (H, W, 3)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1) # (H, W, 1)
            img = np.repeat(img, 3, axis=-1)   # (H, W, 3)
            
        # Case 2: Flattened -> Reshape
        expected_size = 256 * 256 * 3
        if img.ndim == 1 and img.size == expected_size:
             img = img.reshape(256, 256, 3)
             
        # Case 3: Flattened Grayscale -> Reshape -> Expand
        expected_size_gray = 256 * 256
        if img.ndim == 1 and img.size == expected_size_gray:
             img = img.reshape(256, 256)
             img = np.expand_dims(img, axis=-1)
             img = np.repeat(img, 3, axis=-1)

        # 转为 Tensor
        img_tensor = torch.from_numpy(img)
        
        # Permute: (H, W, C) -> (C, H, W)
        img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0

        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        # B. RNA
        rna_tensor = torch.from_numpy(self.rna_data[idx])
        
        # C. Label
        label_tensor = torch.from_numpy(self.protein_data[idx])
        
        return img_tensor, rna_tensor, label_tensor



def create_diagonal_split(full_dataset, batch_size=32, num_workers=0, drop_last=True, plot=True):
    coords = full_dataset.coords
    x = coords[:, 0]
    y = coords[:, 1]

    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    train_mask = x_norm > y_norm
    test_mask = ~train_mask

    indices = np.arange(len(full_dataset))
    train_idx = indices[train_mask]
    val_idx = indices[test_mask]

    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    print("Data Ready (Diagonal Split)!")
    print(f"Train: {len(train_dataset)} spots (Upper Triangle)")
    print(f"Test:  {len(val_dataset)} spots (Lower Triangle)")

    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(x[train_mask], -y[train_mask], s=1, c="red", label="Train (Upper)")
        plt.scatter(x[test_mask], -y[test_mask], s=1, c="blue", label="Test (Lower)")
        plt.legend()
        plt.title("Distribution of Train/Test Sets")
        plt.axis("equal")
        plt.show()

    return train_dataset, val_dataset, train_loader, val_loader


def calculate_pcc(preds, targets):
    pccs = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(preds.shape[1]):
            p = preds[:, i]
            t = targets[:, i]
            if np.std(p) == 0 or np.std(t) == 0:
                pccs.append(0)
            else:
                res = pearsonr(p, t)[0]
                pccs.append(res if not np.isnan(res) else 0)
    return np.mean(pccs)


class PCCLoss(nn.Module):
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
        denom = torch.sqrt(torch.sum(vx**2, dim=0) * torch.sum(vy**2, dim=0)) + self.eps
        return 1.0 - torch.mean(cost / denom)


def train_engine(model_cls, name, train_loader, val_loader, num_proteins, device, epochs=30, lr=1e-4, model_kwargs=None, save_root="models"):
    print(f"\n Training {name} ...")
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

    history = {"train_loss": [], "val_loss": [], "val_pcc": [], "val_rmse": [], "val_r2": []}
    start_epoch = 0
    best_pcc = -1.0

    if os.path.exists(ckpt_path):
        try:
            print(f"Found checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            history = ckpt.get("history", history)
            start_epoch = ckpt["epoch"] + 1
            best_pcc = ckpt.get("best_pcc", best_pcc)
            if start_epoch >= epochs:
                print(f"Already finished {start_epoch} epochs (Target: {epochs}). Skipping.")
                return history
            print(f"Resuming from Epoch {start_epoch + 1} to {epochs}...")
        except Exception as exc:
            print(f"Resume failed ({exc}). Restarting training.")

    if os.path.exists(history_path) and not history["train_loss"]:
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0

        for img, rna, label in tqdm(train_loader, desc=f"[{name}] Ep {epoch + 1}/{epochs}", leave=False):
            img, rna, label = img.to(device), rna.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(img, rna)
            if out.dim() == 3:
                out = out.squeeze(1)

            loss = 0.5 * criterion_mse(out, label) + 0.5 * criterion_pcc(out, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
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

        val_loss /= len(val_loader)
        preds = np.vstack(preds_list)
        labels = np.vstack(labels_list)
        val_pcc = calculate_pcc(preds, labels)
        val_rmse = np.sqrt(mean_squared_error(labels, preds))
        try:
            val_r2 = r2_score(labels, preds)
        except Exception:
            val_r2 = -999.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pcc"].append(float(val_pcc))
        history["val_rmse"].append(float(val_rmse))
        history["val_r2"].append(float(val_r2))

        print(f"   Ep {epoch + 1}: TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f} | PCC={val_pcc:.4f}, RMSE={val_rmse:.4f}")

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

        if val_pcc > best_pcc:
            best_pcc = val_pcc
            torch.save(model.state_dict(), best_path)

        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history, "best_pcc": best_pcc}, ckpt_path)

    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history


def load_histories(model_specs, save_root="models"):
    histories = {}
    for _, name in model_specs:
        hist_path = os.path.join(save_root, name, "history.json")
        if os.path.exists(hist_path):
            with open(hist_path, "r", encoding="utf-8") as f:
                histories[name] = json.load(f)
    return histories


def plot_comparison(histories, save_path="models/comparison_result.png"):
    if not histories:
        print("No history data found for comparison.")
        return

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    for name, hist in histories.items():
        plt.plot(hist["val_loss"], label=f"{name} (Min: {min(hist['val_loss']):.4f})", linewidth=2)
    plt.title("Validation Loss Comparison", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    names = list(histories.keys())
    best_pccs = [max(h["val_pcc"]) for h in histories.values()]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(names)))
    bars = plt.bar(names, best_pccs, color=colors)
    plt.title("Best Validation PCC Comparison", fontsize=14)
    plt.ylabel("Pearson Correlation Coefficient")
    plt.ylim(0, max(best_pccs) * 1.15)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Comparison plot saved to {save_path}")


def load_model_weights(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return model


def evaluate_comprehensive(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for img, rna, label in loader:
            img, rna, label = img.to(device), rna.to(device), label.to(device)
            outputs = model(img, rna)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(label.cpu().numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    mse = mean_squared_error(targets, preds)
    return {
        "PCC": calculate_pcc(preds, targets),
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(targets, preds),
        "R2": r2_score(targets, preds),
    }


def evaluate_models(model_specs, loader, num_proteins, device, model_kwargs=None, save_root="models"):
    metrics_data = []
    model_kwargs = model_kwargs or {}
    for model_cls, model_name in model_specs:
        path = os.path.join(save_root, model_name, "best_model.pth")
        if not os.path.exists(path):
            print(f"Model {model_name} not found, skipping.")
            continue

        print(f"Evaluating: {model_name}...")
        eval_model = model_cls(num_proteins=num_proteins, **model_kwargs).to(device)
        try:
            load_model_weights(eval_model, path, device)
        except Exception as exc:
            print(f"Error loading {model_name}: {exc}")
            continue

        metrics = evaluate_comprehensive(eval_model, loader, device)
        metrics["Model"] = model_name
        metrics_data.append(metrics)
    return pd.DataFrame(metrics_data)


def plot_metrics(df_metrics, save_path="models/comprehensive_comparison.png"):
    if df_metrics.empty:
        print("?? No models evaluated for plotting.")
        return

    sns.set_style("whitegrid")
    _, axes = plt.subplots(2, 2, figsize=(16, 12))
    palette = sns.color_palette("viridis", n_colors=len(df_metrics))

    sns.barplot(data=df_metrics, x="Model", y="PCC", ax=axes[0, 0], palette=palette)
    axes[0, 0].set_title("Pearson Correlation (Higher is Better)", fontsize=14, fontweight="bold")
    axes[0, 0].set_ylim(0, 1.0)
    for i, value in enumerate(df_metrics["PCC"]):
        axes[0, 0].text(i, value + 0.01, f"{value:.4f}", ha="center", fontweight="bold")

    sns.barplot(data=df_metrics, x="Model", y="RMSE", ax=axes[0, 1], palette="magma")
    axes[0, 1].set_title("RMSE (Lower is Better)", fontsize=14, fontweight="bold")
    for i, value in enumerate(df_metrics["RMSE"]):
        axes[0, 1].text(i, value + 0.005, f"{value:.4f}", ha="center", fontweight="bold")

    sns.barplot(data=df_metrics, x="Model", y="R2", ax=axes[1, 0], palette="rocket")
    axes[1, 0].set_title("R2 Score", fontsize=14, fontweight="bold")

    sns.scatterplot(data=df_metrics, x="RMSE", y="PCC", hue="Model", style="Model", s=200, ax=axes[1, 1], palette=palette)
    axes[1, 1].set_title("Efficiency Frontier: RMSE vs PCC", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("RMSE")
    axes[1, 1].set_ylabel("PCC")
    for i in range(df_metrics.shape[0]):
        axes[1, 1].text(df_metrics.RMSE[i], df_metrics.PCC[i] + 0.005, df_metrics.Model[i], fontsize=11, ha="center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"? Comparison plot saved to {save_path}")


def calculate_pcc_single(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return np.corrcoef(y_true, y_pred)[0, 1]


def evaluate_model_per_protein(model_cls, model_name, loader, device, protein_names, model_kwargs=None, save_root="models"):
    model_kwargs = model_kwargs or {}
    path = os.path.join(save_root, model_name, "best_model.pth")
    if not os.path.exists(path):
        print(f"{model_name}: {path}")
        return None

    eval_model = model_cls(num_proteins=len(protein_names), **model_kwargs).to(device)
    try:
        load_model_weights(eval_model, path, device)
    except Exception as exc:
        print(f"? ?? {model_name} ????: {exc}")
        return None

    eval_model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for img, rna, label in loader:
            img, rna, label = img.to(device), rna.to(device), label.to(device)
            outputs = eval_model(img, rna)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(label.cpu().numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    results = []
    for i, protein_name in enumerate(protein_names):
        y_true = targets[:, i]
        y_pred = preds[:, i]
        mse = mean_squared_error(y_true, y_pred)
        results.append({
            "Model": model_name,
            "Protein": protein_name,
            "PCC": calculate_pcc_single(y_true, y_pred),
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_true, y_pred),
        })
    return pd.DataFrame(results)


def evaluate_per_protein_models(model_specs, loader, device, protein_names, model_kwargs=None, save_root="models"):
    all_results = []
    for model_cls, model_name in model_specs:
        result = evaluate_model_per_protein(model_cls, model_name, loader, device, protein_names, model_kwargs=model_kwargs, save_root=save_root)
        if result is not None:
            all_results.append(result)
    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)
