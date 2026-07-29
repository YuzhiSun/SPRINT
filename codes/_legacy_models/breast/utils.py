import gc
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
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


class BreastMultimodalDataset(Dataset):
    def __init__(self, h5ad_path, target_genes=None, target_proteins=None, transform=None):
        print(f"Loading data from: {os.path.basename(h5ad_path)} ...")
        self.adata = sc.read_h5ad(h5ad_path)
        
        # --- 1. RNA Data (Full or Filtered by Intersection) ---
        if target_genes is not None:
            # Normalize file's genes to UPPER for matching
            file_genes_upper = [g.upper() for g in self.adata.var_names]
            gene_map = {g: i for i, g in enumerate(file_genes_upper)}
            
            indices = []
            found_genes = []
            
            for tg in target_genes:
                tg_sup = tg.upper()
                if tg_sup in gene_map:
                    indices.append(gene_map[tg_sup])
                    found_genes.append(tg)
            
            if len(indices) == 0:
                 print(f"Warning: No matching genes found in {os.path.basename(h5ad_path)}! Using all genes.")
                 self.rna_data = self.adata.X
            else:
                self.rna_data = self.adata.X[:, indices]
                self.gene_names = found_genes
        else:
            self.rna_data = self.adata.X
            self.gene_names = self.adata.var_names.tolist()

        # Sparse Handling
        self.is_sparse = scipy.sparse.issparse(self.rna_data)
        if self.is_sparse:
            if not scipy.sparse.isspmatrix_csr(self.rna_data):
                self.rna_data = self.rna_data.tocsr()
        else:
            self.rna_data = self.rna_data.astype(np.float32)

        # --- 2. Image Data (32x32) ---
        if 'spatial_img_crops' in self.adata.obsm:
            img_data = self.adata.obsm['spatial_img_crops']
            if scipy.sparse.issparse(img_data):
                img_data = img_data.toarray()
            img_data = np.array(img_data)
            
            N = img_data.shape[0]
            target_dim = 32 * 32 * 3
            
            if img_data.ndim == 2:
                if img_data.shape[1] == target_dim:
                    img_data = img_data.reshape(N, 32, 32, 3)
                else:
                    side = int(np.sqrt(img_data.shape[1] // 3))
                    img_data = img_data.reshape(N, side, side, 3)
                    
            self.images = img_data
        else:
            print("'spatial_img_crops' not found. Creating Dummy Images.")
            self.images = np.zeros((self.adata.shape[0], 32, 32, 3), dtype=np.uint8)

        # --- 3. Protein Data & Alignment ---
        # A. Load Raw Data
        if 'protein_expression_log' in self.adata.obsm:
             prot_data = self.adata.obsm['protein_expression_log']
        elif 'protein_expression' in self.adata.obsm:
             prot_data = self.adata.obsm['protein_expression']
        else:
             prot_data = np.zeros((self.adata.shape[0], 20))
        
        if hasattr(prot_data, "values"): prot_data = prot_data.values
        if scipy.sparse.issparse(prot_data): prot_data = prot_data.toarray()
        prot_data = prot_data.astype(np.float32)

        # B. Get Protein Names
        if 'protein_names' in self.adata.uns:
            raw_prot_names = [str(x) for x in list(self.adata.uns['protein_names'])]
        else:
            # Fallback: Try to guess or use indices
            raw_prot_names = [str(i) for i in range(prot_data.shape[1])]
            
        # C. Filter & Reorder
        if target_proteins is not None:
            # Normalize names for matching (e.g. UPPER)
            # But protein names are often case-sensitive (e.g. CD4 vs cd4). We try exact match first.
            
            p_map = {p: i for i, p in enumerate(raw_prot_names)}
            # Also add upper case map just in case
            p_map_upper = {p.upper(): i for i, p in enumerate(raw_prot_names)}
            
            valid_indices = []
            
            for tp in target_proteins:
                if tp in p_map:
                    valid_indices.append(p_map[tp])
                elif tp.upper() in p_map_upper:
                    valid_indices.append(p_map_upper[tp.upper()])
                else:
                    # Missing protein in this dataset
                    pass 
            
            if len(valid_indices) > 0:
                self.protein_data = prot_data[:, valid_indices]
            else:
                print(f"⚠️ Warning: No common proteins found for {os.path.basename(h5ad_path)}. Using all.")
                self.protein_data = prot_data
        else:
            self.protein_data = prot_data
        
        self.transform = transform

    def __len__(self):
        return self.rna_data.shape[0]

    def __getitem__(self, idx):
        # Image
        img = self.images[idx]
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        # RNA
        if self.is_sparse:
            rna_vec = self.rna_data[idx].toarray().flatten()
        else:
            rna_vec = self.rna_data[idx]
        rna_tensor = torch.from_numpy(rna_vec)
        
        # Protein
        prot_tensor = torch.from_numpy(self.protein_data[idx])
        
        return img_tensor, rna_tensor, prot_tensor


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
    return float(np.mean(pccs))


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


def instantiate_model(model_cls, num_proteins, num_genes, device):
    try:
        return model_cls(num_proteins=num_proteins, num_genes=num_genes).to(device)
    except TypeError:
        return model_cls(num_proteins=num_proteins).to(device)


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
    save_root="models_breast",
    resume=False,
    lr_decay_step=5,
    lr_decay_factor=0.9,
    lr_min=1e-7,
    grad_clip_norm=5.0,
):
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
    history = {"train_loss": [], "val_loss": [], "val_pcc": [], "val_rmse": [], "val_r2": [], "lr": []}
    start_epoch = 0
    best_pcc = -1.0
    if resume and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get("model", ckpt.get("model_state_dict", ckpt)))
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            history = ckpt.get("history", history)
            history.setdefault("lr", [])
            start_epoch = ckpt.get("epoch", -1) + 1
            best_pcc = ckpt.get("best_pcc", best_pcc)
            if start_epoch >= epochs:
                return history
        except Exception as exc:
            print(f"Resume failed for {name}: {exc}")
    elif os.path.exists(ckpt_path):
        print(f"{name}: resume=False, ignoring existing checkpoint {ckpt_path}")

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
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{name} epoch {epoch + 1}: non-finite training loss detected. "
                    f"Try a smaller lr; current lr={optimizer.param_groups[0]['lr']:.2e}."
                )
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

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
                if not torch.isfinite(out).all() or not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"{name} epoch {epoch + 1}: validation produced NaN/Inf values. "
                        f"Try a smaller lr; current lr={optimizer.param_groups[0]['lr']:.2e}."
                    )
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
        current_lr = float(optimizer.param_groups[0]["lr"])
        history["lr"].append(current_lr)
        lr_message = ""
        if lr_decay_step and (epoch + 1) % lr_decay_step == 0:
            new_lr = max(current_lr * lr_decay_factor, lr_min)
            if new_lr < current_lr:
                for group in optimizer.param_groups:
                    group["lr"] = max(float(group["lr"]) * lr_decay_factor, lr_min)
                lr_message = f", lr_decay={current_lr:.2e}->{new_lr:.2e}"
        print(f"{name} epoch {epoch + 1}: loss={train_loss:.4f}, val_pcc={val_pcc:.4f}, val_rmse={val_rmse:.4f}, lr={current_lr:.2e}{lr_message}")
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


def load_model_weights(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    return model


def load_histories(model_specs, save_root="models_breast"):
    histories = {}
    for _, name in model_specs:
        path = os.path.join(save_root, name, "history.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                histories[name] = json.load(f)
    return histories


def evaluate_comprehensive(model, loader, device):
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
    mse_raw = mean_squared_error(targets, preds, multioutput="raw_values")
    return {"PCC": calculate_pcc(preds, targets), "RMSE": float(np.sqrt(np.mean(mse_raw))), "MAE": float(mean_absolute_error(targets, preds)), "R2": float(r2_score(targets, preds)), "RMSE_Per_Target": np.sqrt(mse_raw)}


def plot_comparison(histories, save_path):
    if not histories:
        print("No history data found.")
        return
    plt.figure(figsize=(14, 8))
    for name, hist in histories.items():
        if hist.get("val_pcc"):
            plt.plot(hist["val_pcc"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation PCC")
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
