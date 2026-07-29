import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

from model import SelfAttentionMSIModel
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable



def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MouseBrainMSIDataset(Dataset):
    def __init__(self, h5ad_path, metabolite_names=None, target_mean=None, target_std=None, image_norm=True):
        self.adata = sc.read_h5ad(h5ad_path)
        self.images = self.adata.obsm["spatial_img_crops"]
        self.rna_data = self.adata.X
        self.is_sparse = sp.issparse(self.rna_data)
        if self.is_sparse and not sp.isspmatrix_csr(self.rna_data):
            self.rna_data = self.rna_data.tocsr()
        elif not self.is_sparse:
            self.rna_data = np.asarray(self.rna_data, dtype=np.float32)

        raw_names = [str(x) for x in list(self.adata.uns["metabolite_names"])]
        raw_y = self.adata.obsm["metabolite_expression_log"]
        raw_y = raw_y.values if hasattr(raw_y, "values") else raw_y
        raw_y = np.asarray(raw_y, dtype=np.float32)

        if metabolite_names is None:
            metabolite_names = raw_names
        idx = [raw_names.index(name) for name in metabolite_names]
        self.metabolite_names = list(metabolite_names)
        self.y_original = raw_y[:, idx]

        if target_mean is None:
            target_mean = np.zeros(self.y_original.shape[1], dtype=np.float32)
        if target_std is None:
            target_std = np.ones(self.y_original.shape[1], dtype=np.float32)
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_std = np.asarray(target_std, dtype=np.float32)
        self.y = (self.y_original - self.target_mean) / self.target_std
        self.image_norm = image_norm

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        # uint8 RGB 图像转成 float tensor，并按 ResNet50 预训练规范归一化。
        img = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float() / 255.0
        if self.image_norm:
            img = (img - IMAGE_MEAN) / IMAGE_STD

        if self.is_sparse:
            rna = self.rna_data[idx].toarray().ravel()
        else:
            rna = self.rna_data[idx]
        # RNA 已在预处理阶段按 tonsil 流程 normalize_total + log1p，这里只转成 tensor。
        rna = torch.from_numpy(np.asarray(rna, dtype=np.float32))
        y = torch.from_numpy(self.y[idx])
        return img, rna, y


class StablePCCLoss(nn.Module):
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
    vals = []
    for i in range(preds.shape[1]):
        p = preds[:, i]
        t = targets[:, i]
        if np.std(p) == 0 or np.std(t) == 0:
            vals.append(0.0)
        else:
            r = pearsonr(p, t)[0]
            vals.append(0.0 if np.isnan(r) else float(r))
    return float(np.mean(vals))


def evaluate(model, loader, mean, std):
    model.eval()
    preds_z, targets_z = [], []
    with torch.no_grad():
        for img, rna, y in loader:
            out = model(img.to(DEVICE), rna.to(DEVICE))
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
    image_params = []
    other_params = []
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
    # Linear warmup 后接 cosine decay。LambdaLR 会同时作用到所有 param groups，
    # 因此 image tower 仍保持 lr * image_lr_factor 的相对较小学习率。
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
    return {f"lr_group_{i}": float(group["lr"]) for i, group in enumerate(optimizer.param_groups)}


def train_one(args, scheme, train_loader, val_loader, num_genes, num_metabolites, target_mean, target_std, out_dir):
    model = SelfAttentionMSIModel(scheme, num_genes, num_metabolites, freeze_image=args.freeze_image).to(DEVICE)
    optimizer = make_optimizer(model, args.lr, args.image_lr_factor, args.weight_decay)
    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr_factor=args.min_lr_factor,
    )
    mse = nn.MSELoss()
    pcc = StablePCCLoss()
    history = {"train_loss": [], "val_pcc": [], "val_rmse": [], "val_r2": [], "lr_group_0": [], "lr_group_1": []}
    best_pcc = -1.0
    best_path = out_dir / "best_model.pth"
    ckpt_path = out_dir / "checkpoint.pth"
    start_epoch = 0

    # 默认自动续训：同一个 tag 和 scheme 下若存在 checkpoint.pth，就从下一轮 epoch 继续。
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        history = ckpt.get("history", history)
        best_pcc = float(ckpt.get("best_pcc", max(history.get("val_pcc", [-1.0]))))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"{scheme}: resume from epoch {start_epoch + 1}, best PCC={best_pcc:.4f}", flush=True)

    if start_epoch >= args.epochs:
        print(f"{scheme}: checkpoint already reached {start_epoch} epochs; skip training.", flush=True)
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        final = evaluate(model, val_loader, target_mean, target_std)
        return model, final, history

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = 0.0
        batch_iter = tqdm(
            train_loader,
            desc=f"{scheme} epoch {epoch + 1}/{args.epochs}",
            unit="batch",
            leave=False,
        )
        for step, (img, rna, y) in enumerate(batch_iter, start=1):
            img, rna, y = img.to(DEVICE), rna.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(img, rna)
            loss = args.mse_weight * mse(out, y) + args.pcc_weight * pcc(out, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{scheme} non-finite loss at epoch {epoch+1}, step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += loss.item()
            if hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(loss=f"{total / step:.4f}")
        metrics = evaluate(model, val_loader, target_mean, target_std)
        history["train_loss"].append(total / len(train_loader))
        history["val_pcc"].append(metrics["pcc"])
        history["val_rmse"].append(metrics["rmse"])
        history["val_r2"].append(metrics["r2"])
        current_lrs = get_lr_dict(optimizer)
        for key, value in current_lrs.items():
            history.setdefault(key, []).append(value)
        print(
            f"{scheme} epoch {epoch+1}/{args.epochs}: "
            f"loss={history['train_loss'][-1]:.4f}, PCC={metrics['pcc']:.4f}, "
            f"RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}, "
            f"lr0={current_lrs.get('lr_group_0', float('nan')):.2e}, "
            f"lr1={current_lrs.get('lr_group_1', float('nan')):.2e}",
            flush=True,
        )
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_pcc": max(best_pcc, metrics["pcc"]),
            "history": history,
            "target_mean": target_mean,
            "target_std": target_std,
            "args": vars(args),
            "scheme": scheme,
        }
        torch.save(state, ckpt_path)
        if metrics["pcc"] > best_pcc:
            best_pcc = metrics["pcc"]
            torch.save(state, best_path)
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        scheduler.step()

    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final = evaluate(model, val_loader, target_mean, target_std)
    return model, final, history


def save_per_metabolite(preds, targets, names, out_path):
    rows = []
    for i, name in enumerate(names):
        p = preds[:, i]
        t = targets[:, i]
        if np.std(p) == 0 or np.std(t) == 0:
            pcc = 0.0
        else:
            r = pearsonr(p, t)[0]
            pcc = 0.0 if np.isnan(r) else float(r)
        rows.append(
            {
                "Metabolite": name,
                "PCC": pcc,
                "RMSE": float(np.sqrt(mean_squared_error(t, p))),
                "R2": float(r2_score(t, p)),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


