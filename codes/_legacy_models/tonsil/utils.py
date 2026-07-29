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


class TonsilMultimodalDataset(Dataset):
    def __init__(self, h5ad_path, target_proteins=None, transform=None):
        """
        target_proteins: List of protein names to filter. If None, use all.
        """
        print(f"Loading data from: {h5ad_path} ...")
        self.adata = sc.read_h5ad(h5ad_path)
        
        # 1. 图像数据 (Input A)
        self.images = self.adata.obsm['spatial_img_crops']
        
        # 2. RNA 数据 (Input B)
        self.rna_data = self.adata.X
        self.gene_names = self.adata.var_names.tolist()
            
        # 优化内存：检查是否为稀疏矩阵
        self.is_sparse = scipy.sparse.issparse(self.rna_data)
        if self.is_sparse:
            if not scipy.sparse.isspmatrix_csr(self.rna_data):
                self.rna_data = self.rna_data.tocsr()
        else:
            self.rna_data = self.rna_data.astype(np.float32)

        # 3. 蛋白数据 (Target) & 4. 元数据 (Protein Names)
        raw_protein_data = self.adata.obsm['protein_expression_log']
        if hasattr(raw_protein_data, "values"):
            raw_protein_data = raw_protein_data.values
        raw_protein_data = raw_protein_data.astype(np.float32)
        
        # Consistent naming logic (Fix for inconsistent protein names)
        # Use simple stringified index if no names found, to match main block fallback
        if 'protein_names' in self.adata.uns:
            raw_protein_names = list(self.adata.uns['protein_names'])
            # Ensure they are strings
            raw_protein_names = [str(x) for x in raw_protein_names]
        else:
            # Fallback must match the main block fallback logic!
            # If main block finds "0", "1", then this must also be "0", "1"
            # Switch to simple str(i) which is safer if uns is missing
            raw_protein_names = [str(i) for i in range(raw_protein_data.shape[1])]
            
        # --- Filter Logic ---
        if target_proteins is not None:
            print(f" Filtering proteins: intersection of {len(raw_protein_names)} and target {len(target_proteins)}...")
            
            # Find indices of target_proteins in this dataset
            valid_indices = []
            valid_names = []
            
            # Create a lookup for O(1) access
            name_to_idx = {name: i for i, name in enumerate(raw_protein_names)}
            
            for t_name in target_proteins:
                if t_name in name_to_idx:
                    valid_indices.append(name_to_idx[t_name])
                    valid_names.append(t_name)
                else:
                    # Provide clearer warning
                    print(f" Warning: Target protein '{t_name}' not found in dataset names. (Sample existing names: {raw_protein_names[:3]}...)")
            
            if len(valid_indices) == 0:
                print(f" FATAL: No common proteins found.")
                print(f"      Target sample: {target_proteins[:5]}")
                print(f"Dataset sample: {raw_protein_names[:5]}")
                raise ValueError("No common proteins found between target list and dataset!")
                
            self.protein_data = raw_protein_data[:, valid_indices]
            self.protein_names = valid_names
            print(f" Filtered to {len(self.protein_names)} proteins.")
        else:
            self.protein_data = raw_protein_data
            self.protein_names = raw_protein_names
            print(f"Using all {len(self.protein_names)} proteins.")

        # 5. 空间坐标
        self.coords = self.adata.obsm['spatial']
        self.transform = transform

    def __len__(self):
        return self.rna_data.shape[0]

    def __getitem__(self, idx):
        # A. Image
        img = self.images[idx]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        # B. RNA
        if self.is_sparse:
            rna_vector = self.rna_data[idx].toarray().flatten()
        else:
            rna_vector = self.rna_data[idx]
        rna_tensor = torch.from_numpy(rna_vector.astype(np.float32))
        
        # C. Label
        label_tensor = torch.from_numpy(self.protein_data[idx])
        
        return img_tensor, rna_tensor, label_tensor



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


def train_engine(model_cls, name, train_loader, val_loader, num_proteins, device, epochs=30, lr=1e-4, model_kwargs=None, save_root="models"):
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
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get("model", ckpt.get("model_state_dict", ckpt)))
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            history = ckpt.get("history", history)
            start_epoch = ckpt.get("epoch", -1) + 1
            best_pcc = ckpt.get("best_pcc", best_pcc)
            if start_epoch >= epochs:
                return history
        except Exception as exc:
            print(f"Resume failed for {name}: {exc}")

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
        print(f"{name} epoch {epoch + 1}: loss={train_loss:.4f}, val_pcc={val_pcc:.4f}, val_rmse={val_rmse:.4f}")
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


def load_model_weights(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    return model


def load_histories(model_specs, save_root="models"):
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


# ============================================================
# Image Resolution Auto-Detection: Scheme A (high-res) vs Scheme C (low-res)
# ============================================================
# Reference mapping from existing experiments:
#   Spleen:  14x14 px  -> Scheme C (low-res)
#   Breast:  32x32 px  -> Scheme C (low-res)
#   Tonsil:  256x256 px -> Scheme A (high-res)
#   Brain:   256x256 px -> Scheme A (high-res)
#   MSI:     256x256 px -> Scheme A (high-res)
#
# Rationale:
#   - Scheme A (ImageProcessor_SchemeA / ImageProcessor_A): spatial flatten into
#     256 tokens — preserves spatial detail, suitable for high-res images.
#   - Scheme C (ImageProcessor_SchemeC / ImageProcessor_C): global average pool
#     into 4 tokens — discards spatial structure, suitable for low-res images
#     where spatial detail is limited and pooling avoids noise.
#
# Threshold: patch size >= 128 px → high-res (Scheme A); else low-res (Scheme C)

_IMAGE_RESOLUTION_THRESHOLD = 128


def detect_image_resolution_from_data(h5ad_path):
    """
    Detect whether image patches in a dataset are high-resolution or
    low-resolution by inspecting 'spatial_img_crops' in the .h5ad file.

    Parameters
    ----------
    h5ad_path : str or Path
        Path to the .h5ad file.

    Returns
    -------
    (scheme_label, patch_size) : (str, int)
        scheme_label is either "A" (high-res) or "C" (low-res).
        patch_size is the detected image patch size in pixels.
    """
    ad = sc.read_h5ad(h5ad_path, backed="r")
    crops = ad.obsm.get("spatial_img_crops", None)
    if crops is None:
        raise KeyError(
            f"{h5ad_path} does not contain 'spatial_img_crops' in .obsm. "
            f"Available keys: {list(ad.obsm.keys())}"
        )
    # crops shape: (n_spots, H, W, C) or (n_spots, H, W)
    shape = crops.shape
    patch_size = shape[1] if len(shape) >= 2 else 0
    del ad
    if patch_size >= _IMAGE_RESOLUTION_THRESHOLD:
        scheme = "A"
    else:
        scheme = "C"
    return scheme, int(patch_size)


def select_image_processor_scheme(h5ad_path=None, dataset=None, manual_scheme=None):
    """
    Determine which image processor scheme to use for training.

    Detection logic (based on known experimental results):
        - Patch size >= 128 px → Scheme A (high-res, spatial flatten, 256 tokens)
        - Patch size <  128 px → Scheme C (low-res, global pool, 4 tokens)

    Parameters
    ----------
    h5ad_path : str or Path, optional
        Path to a training .h5ad file. Used when `dataset` is not provided.
    dataset : Dataset instance, optional
        A loaded PyTorch Dataset (e.g. TonsilMultimodalDataset).
        Must have ``.images`` attribute (shape: n_spots x H x W x C).
    manual_scheme : str or None, optional
        Force a specific scheme: "A", "C", or None (auto-detect).
        Overrides all automatic detection.

    Returns
    -------
    scheme : str
        "A" or "C".

    Examples
    --------
    # Auto-detect from h5ad file
    scheme = select_image_processor_scheme(h5ad_path="Tonsil_1_Final.h5ad")

    # Auto-detect from dataset
    train_dataset = TonsilMultimodalDataset(...)
    scheme = select_image_processor_scheme(dataset=train_dataset)

    # Manual override
    scheme = select_image_processor_scheme(h5ad_path="...", manual_scheme="A")
    """
    if manual_scheme is not None:
        manual_scheme = str(manual_scheme).upper().strip()
        if manual_scheme not in ("A", "C"):
            raise ValueError(f"manual_scheme must be 'A' or 'C', got '{manual_scheme}'")
        return manual_scheme

    if dataset is not None:
        images = getattr(dataset, "images", None)
        if images is not None:
            shape = images.shape
            patch_size = int(shape[1]) if len(shape) >= 2 else 0
            scheme = "A" if patch_size >= _IMAGE_RESOLUTION_THRESHOLD else "C"
            return scheme

    if h5ad_path is not None:
        scheme, _ = detect_image_resolution_from_data(h5ad_path)
        return scheme

    raise ValueError(
        "Must provide at least one of: h5ad_path, dataset, or manual_scheme"
    )
