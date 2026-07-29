"""
PyTorch Dataset classes for SPRINT multi-modal spatial data.

Supported experiment types
--------------------------
- Brain (256px, Scheme A2)     : BrainMultimodalDataset + diagonal split
- Spleen (14px, Scheme C2)     : SpleenMultimodalDataset  + cross-sample
- Breast (32px, Scheme C2)     : BreastMultimodalDataset  + cross-species
- Tonsil (256px, Scheme A2)    : TonsilMultimodalDataset  + cross-sample + auto-scheme
- MSI Mouse Brain (256px, A2)  : MouseBrainMSIDataset     + z-score normalization
"""

import os
from pathlib import Path

import numpy as np
import scipy.sparse
import torch
from torch.utils.data import DataLoader, Dataset, Subset


# ===========================================================================
# Internal helpers
# ===========================================================================

def _resolve_sparse(matrix):
    """Ensure sparse matrix is CSR; return dense if already dense."""
    if scipy.sparse.issparse(matrix):
        if not scipy.sparse.isspmatrix_csr(matrix):
            return matrix.tocsr()
        return matrix
    return np.asarray(matrix, dtype=np.float32)


def _resolve_protein_data(adata, target_proteins=None):
    """Extract protein expression matrix and optionally filter by target names."""
    protein = adata.obsm.get("protein_expression_log", None)
    if protein is None:
        protein = adata.obsm.get("protein_expression", None)
    if protein is None:
        raise KeyError(
            "h5ad must contain 'protein_expression_log' or 'protein_expression' in .obsm"
        )

    if hasattr(protein, "values"):
        protein = protein.values
    if scipy.sparse.issparse(protein):
        protein = protein.toarray()
    protein = np.asarray(protein, dtype=np.float32)

    # Protein names
    if "protein_names" in adata.uns:
        protein_names = [str(x) for x in list(adata.uns["protein_names"])]
    else:
        protein_names = [str(i) for i in range(protein.shape[1])]

    if target_proteins is not None:
        name_to_idx = {n: i for i, n in enumerate(protein_names)}
        indices = []
        for n in target_proteins:
            if n in name_to_idx:
                indices.append(name_to_idx[n])
        if indices:
            protein = protein[:, indices]
            protein_names = [n for n in target_proteins if n in name_to_idx]

    return protein, protein_names


def _prepare_image(img_arr):
    """Convert raw image array → torch tensor (C,H,W), float32, 0-1."""
    img = np.asarray(img_arr)
    if img.ndim == 1:
        total = img.size
        for side in [256, 32, 14]:
            if total == side * side * 3:
                img = img.reshape(side, side, 3)
                break
            if total == side * side:
                img = np.repeat(img.reshape(side, side, 1), 3, axis=-1)
                break
        else:
            side = int(np.sqrt(total / 3))
            img = img.reshape(side, side, 3)
    elif img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)

    tensor = torch.from_numpy(img).float()
    if tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)
    return tensor / 255.0


# ===========================================================================
# Dataset: Brain (256px, Scheme A2)
# ===========================================================================

class BrainMultimodalDataset(Dataset):
    """Brain Spatial Transcriptomics + H&E + Protein dataset (256×256 crops)."""

    def __init__(self, h5ad_path, transform=None):
        import scanpy as sc
        print(f"Loading data from: {h5ad_path} ...")
        self.adata = sc.read_h5ad(h5ad_path)

        # Images
        if "spatial_img_crops" in self.adata.obsm:
            self.images = self.adata.obsm["spatial_img_crops"]
        else:
            raise KeyError("spatial_img_crops not found in obsm")
        if scipy.sparse.issparse(self.images):
            self.images = self.images.toarray()
        print(f"Raw Images Shape: {self.images.shape}")

        # RNA
        print("Using ALL Genes (No HVG filter).")
        self.rna_data = _resolve_sparse(self.adata.X)
        self.gene_names = self.adata.var_names.tolist()
        print(f"RNA Input Shape: {self.rna_data.shape}")

        # Proteins
        self.protein_data, self.protein_names = _resolve_protein_data(self.adata)
        self.coords = np.asarray(self.adata.obsm["spatial"])
        self.transform = transform

    def __len__(self):
        return self.rna_data.shape[0]

    def __getitem__(self, idx):
        img_tensor = _prepare_image(self.images[idx])
        if self.transform:
            img_tensor = self.transform(img_tensor)
        if scipy.sparse.issparse(self.rna_data):
            rna_vec = self.rna_data[idx].toarray().flatten().astype(np.float32)
        else:
            rna_vec = np.asarray(self.rna_data[idx], dtype=np.float32)
        return (
            img_tensor,
            torch.from_numpy(rna_vec),
            torch.from_numpy(self.protein_data[idx]),
        )


# ===========================================================================
# Dataset: Spleen (14px, Scheme C2)
# ===========================================================================

class SpleenMultimodalDataset(Dataset):
    """Spleen Spatial Transcriptomics + H&E + Protein dataset (14×14 crops)."""

    def __init__(self, h5ad_path, target_proteins=None, transform=None):
        import scanpy as sc
        print(f"Loading data from: {h5ad_path} ...")
        self.adata = sc.read_h5ad(h5ad_path)

        # Images
        if "spatial_img_crops" in self.adata.obsm:
            img_data = self.adata.obsm["spatial_img_crops"]
            if hasattr(img_data, "toarray"):
                img_data = img_data.toarray()
            elif hasattr(img_data, "todense"):
                img_data = img_data.todense()
            img_data = np.array(img_data)
            N = img_data.shape[0]
            if img_data.ndim == 2:
                dim = img_data.shape[1]
                if dim == 14 * 14 * 3:
                    img_data = img_data.reshape(N, 14, 14, 3)
                elif dim == 14 * 14:
                    img_data = img_data.reshape(N, 14, 14)
                    img_data = np.stack([img_data] * 3, axis=-1)
                else:
                    side = int(np.sqrt(dim // 3))
                    if side * side * 3 == dim:
                        img_data = img_data.reshape(N, side, side, 3)
                    else:
                        img_data = np.zeros((N, 14, 14, 3), dtype=np.uint8)
            elif img_data.ndim == 3:
                img_data = np.stack([img_data] * 3, axis=-1)
            self.images = img_data
        else:
            print("'spatial_img_crops' not found. Creating dummy images.")
            self.images = np.zeros((self.adata.shape[0], 14, 14, 3), dtype=np.uint8)

        # RNA
        self.rna_data = _resolve_sparse(self.adata.X)
        self.gene_names = self.adata.var_names.tolist()

        # Proteins
        self.protein_data, self.protein_names = _resolve_protein_data(
            self.adata, target_proteins
        )
        self.transform = transform

    def __len__(self):
        return self.rna_data.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.ndim == 1:
            side = int(np.sqrt(img.shape[0] // 3))
            img = img.reshape(side, side, 3)
        img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        if self.transform:
            img_tensor = self.transform(img_tensor)
        if scipy.sparse.issparse(self.rna_data):
            rna_vec = self.rna_data[idx].toarray().flatten().astype(np.float32)
        else:
            rna_vec = np.asarray(self.rna_data[idx], dtype=np.float32)
        return (
            img_tensor,
            torch.from_numpy(rna_vec),
            torch.from_numpy(self.protein_data[idx]),
        )


# ===========================================================================
# Dataset: Breast (32px, Scheme C2, cross-species)
# ===========================================================================

class BreastMultimodalDataset(Dataset):
    """Breast Cancer Spatial + H&E + Protein dataset (32×32 crops).
    
    Supports cross-species gene/protein alignment via target_genes and target_proteins.
    """

    def __init__(self, h5ad_path, target_genes=None, target_proteins=None, transform=None):
        import scanpy as sc
        print(f"Loading data from: {os.path.basename(h5ad_path)} ...")
        self.adata = sc.read_h5ad(h5ad_path)

        # RNA with gene subsetting
        if target_genes is not None:
            file_genes_upper = [g.upper() for g in self.adata.var_names]
            gene_map = {g: i for i, g in enumerate(file_genes_upper)}
            indices = []
            for tg in target_genes:
                if tg.upper() in gene_map:
                    indices.append(gene_map[tg.upper()])
            if indices:
                self.rna_data = self.adata.X[:, indices]
            else:
                print("Warning: No matching genes found. Using all.")
                self.rna_data = self.adata.X
        else:
            self.rna_data = self.adata.X

        self.is_sparse = scipy.sparse.issparse(self.rna_data)
        if self.is_sparse:
            if not scipy.sparse.isspmatrix_csr(self.rna_data):
                self.rna_data = self.rna_data.tocsr()
        else:
            self.rna_data = np.asarray(self.rna_data, dtype=np.float32)

        # Images (32px)
        if "spatial_img_crops" in self.adata.obsm:
            img_data = self.adata.obsm["spatial_img_crops"]
            if scipy.sparse.issparse(img_data):
                img_data = img_data.toarray()
            img_data = np.array(img_data)
            N = img_data.shape[0]
            if img_data.ndim == 2:
                if img_data.shape[1] == 32 * 32 * 3:
                    img_data = img_data.reshape(N, 32, 32, 3)
                else:
                    side = int(np.sqrt(img_data.shape[1] // 3))
                    img_data = img_data.reshape(N, side, side, 3)
            self.images = img_data
        else:
            self.images = np.zeros((self.adata.shape[0], 32, 32, 3), dtype=np.uint8)

        # Proteins
        self.protein_data, self.protein_names = _resolve_protein_data(
            self.adata, target_proteins
        )
        self.transform = transform

    def __len__(self):
        return self.rna_data.shape[0]

    def __getitem__(self, idx):
        img_tensor = (
            torch.from_numpy(self.images[idx])
            .float()
            .permute(2, 0, 1)
            / 255.0
        )
        if self.transform:
            img_tensor = self.transform(img_tensor)
        if self.is_sparse:
            rna_vec = self.rna_data[idx].toarray().flatten()
        else:
            rna_vec = self.rna_data[idx]
        return (
            img_tensor,
            torch.from_numpy(rna_vec),
            torch.from_numpy(self.protein_data[idx]),
        )


# ===========================================================================
# Dataset: Tonsil (256px, Scheme A2, cross-sample)
# ===========================================================================

class TonsilMultimodalDataset(Dataset):
    """Tonsil Spatial + H&E + Protein dataset (256×256 crops)."""

    def __init__(self, h5ad_path, target_proteins=None, transform=None):
        import scanpy as sc
        print(f"Loading data from: {h5ad_path} ...")
        self.adata = sc.read_h5ad(h5ad_path)

        # Images
        self.images = self.adata.obsm["spatial_img_crops"]

        # RNA
        self.rna_data = _resolve_sparse(self.adata.X)
        self.gene_names = self.adata.var_names.tolist()

        # Proteins
        self.protein_data, self.protein_names = _resolve_protein_data(
            self.adata, target_proteins
        )
        if target_proteins is not None:
            print(f"Filtering proteins → {len(self.protein_names)} proteins.")

        self.coords = np.asarray(self.adata.obsm["spatial"])
        self.transform = transform

    def __len__(self):
        return self.rna_data.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.transform:
            img_tensor = self.transform(img_tensor)
        if scipy.sparse.issparse(self.rna_data):
            rna_vec = self.rna_data[idx].toarray().flatten().astype(np.float32)
        else:
            rna_vec = np.asarray(self.rna_data[idx], dtype=np.float32)
        return (
            img_tensor,
            torch.from_numpy(rna_vec),
            torch.from_numpy(self.protein_data[idx]),
        )


# ===========================================================================
# Dataset: MSI Mouse Brain (Z-score normalized metabolites)
# ===========================================================================

class MouseBrainMSIDataset(Dataset):
    """MSI Mouse Brain: Spatial Transcriptomics + H&E + Metabolite dataset.

    Metabolite targets are z-score normalized using precomputed mean/std.
    """

    def __init__(
        self, h5ad_path, metabolite_names=None, target_mean=None,
        target_std=None, image_norm=True,
    ):
        import scanpy as sc
        self.adata = sc.read_h5ad(h5ad_path)
        self.images = self.adata.obsm["spatial_img_crops"]
        self.rna_data = _resolve_sparse(self.adata.X)

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

        self.IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float() / 255.0
        if self.image_norm:
            img = (img - self.IMAGE_MEAN) / self.IMAGE_STD

        if scipy.sparse.issparse(self.rna_data):
            rna = self.rna_data[idx].toarray().ravel()
        else:
            rna = self.rna_data[idx]
        rna = torch.from_numpy(np.asarray(rna, dtype=np.float32))
        y = torch.from_numpy(self.y[idx])
        return img, rna, y


# ===========================================================================
# Spatial diagonal train/test split
# ===========================================================================

def create_diagonal_split(
    dataset, batch_size=32, num_workers=0, drop_last=True, plot=True
):
    """Split dataset by spatial diagonal (upper-triangle = train, lower = test)."""
    coords = dataset.coords
    x, y = coords[:, 0], coords[:, 1]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
    train_mask = x_norm > y_norm
    test_mask = ~train_mask

    indices = np.arange(len(dataset))
    train_idx = indices[train_mask]
    test_idx = indices[test_mask]

    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=drop_last,
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )

    print("Data Ready (Diagonal Split)!")
    print(f"Train: {len(train_idx)} spots (Upper Triangle)")
    print(f"Test:  {len(test_idx)} spots (Lower Triangle)")

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.scatter(x[train_mask], -y[train_mask], s=1, c="red", label="Train (Upper)")
        plt.scatter(x[test_mask], -y[test_mask], s=1, c="blue", label="Test (Lower)")
        plt.legend()
        plt.title("Distribution of Train/Test Sets")
        plt.axis("equal")
        plt.show()

    return train_subset, test_subset, train_loader, test_loader


# ===========================================================================
# Shared gene / protein utils for cross-species experiments
# ===========================================================================

def shared_uppercase_genes(h5ad_a, h5ad_b):
    """Return sorted list of genes shared between two .h5ad files (case-insensitive)."""
    import scanpy as sc
    a = sc.read_h5ad(h5ad_a, backed="r")
    b = sc.read_h5ad(h5ad_b, backed="r")
    genes = sorted(set(g.upper() for g in a.var_names) & set(g.upper() for g in b.var_names))
    a.file.close()
    b.file.close()
    return genes


def shared_proteins(h5ad_a, h5ad_b):
    """Return sorted list of protein names shared between two .h5ad files."""
    names_a = _protein_names_from_file(h5ad_a)
    names_b = _protein_names_from_file(h5ad_b)
    return sorted(set(names_a) & set(names_b))


def _protein_names_from_file(h5ad_path):
    """Read protein names from .h5ad obsm group."""
    try:
        import h5py
        with h5py.File(h5ad_path, "r") as f:
            group = f.get("obsm/protein_expression_log") or f.get("obsm/protein_expression")
            if group is None:
                return []
            order = group.attrs.get("column-order")
            if order is not None:
                return [x.decode() if isinstance(x, bytes) else str(x) for x in list(order)]
            return [str(k) for k in group.keys() if k != "_index"]
    except Exception:
        return []


# ===========================================================================
# Image resolution auto-detection (for tonsil-like workflows)
# ===========================================================================

_IMAGE_RESOLUTION_THRESHOLD = 128


def detect_image_resolution(h5ad_path=None, dataset=None, manual_scheme=None):
    """Determine image processor scheme (A or C) based on patch resolution.

    - Patch size >= 128 px → Scheme A (high-res, spatial flatten, 256 tokens)
    - Patch size <  128 px → Scheme C (low-res, global pool)

    Parameters
    ----------
    h5ad_path : str, optional
    dataset : Dataset, optional (must have .images attribute)
    manual_scheme : str, optional — "A" or "C" to override auto-detection.

    Returns
    -------
    scheme : str — "A" or "C".
    """
    if manual_scheme is not None:
        manual_scheme = str(manual_scheme).upper().strip()
        if manual_scheme not in ("A", "C"):
            raise ValueError(f"manual_scheme must be 'A' or 'C', got '{manual_scheme}'")
        return manual_scheme

    if dataset is not None:
        images = getattr(dataset, "images", None)
        if images is not None:
            patch_size = int(images.shape[1]) if len(images.shape) >= 2 else 0
            return "A" if patch_size >= _IMAGE_RESOLUTION_THRESHOLD else "C"

    if h5ad_path is not None:
        import scanpy as sc
        ad = sc.read_h5ad(h5ad_path, backed="r")
        crops = ad.obsm.get("spatial_img_crops")
        if crops is None:
            raise KeyError(f"No 'spatial_img_crops' in {h5ad_path}")
        patch_size = int(crops.shape[1]) if len(crops.shape) >= 2 else 0
        del ad
        return "A" if patch_size >= _IMAGE_RESOLUTION_THRESHOLD else "C"

    raise ValueError("Must provide h5ad_path, dataset, or manual_scheme")


# Legacy alias
select_image_processor_scheme = detect_image_resolution
