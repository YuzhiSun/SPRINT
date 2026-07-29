"""General utilities for SPRINT."""

import os
import subprocess
import random

import numpy as np
import torch


def set_seed(seed=42):
    """Set random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return the best available torch device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def auto_select_gpu():
    """Select the least-utilised GPU (via nvidia-smi) and return torch device.

    Falls back to ``get_device()`` if nvidia-smi is unavailable.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        gpu_rows = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            idx, used, total = int(parts[0]), int(parts[1]), int(parts[2])
            gpu_rows.append(
                {
                    "index": idx,
                    "used_mb": used,
                    "total_mb": total,
                    "ratio": used / max(total, 1),
                }
            )

        if not gpu_rows:
            return get_device()

        best = min(gpu_rows, key=lambda r: (r["used_mb"], r["ratio"], r["index"]))
        print("Detected GPUs:")
        for r in gpu_rows:
            mark = " ★" if r["index"] == best["index"] else ""
            print(
                f"  cuda:{r['index']}  {r['used_mb']:>5} / {r['total_mb']} MB  "
                f"({r['ratio']:.1%}){mark}"
            )

        return torch.device(f"cuda:{best['index']}")
    except Exception as exc:
        print(f"nvidia-smi unavailable ({exc}); falling back to default CUDA.")
        return get_device()


def generate_demo_data(input_path, output_path="datas/demo/demo_brain.h5ad",
                       n_spots=200, random_seed=42):
    """Generate a small demo .h5ad dataset for quick pipeline testing.

    Parameters
    ----------
    input_path : str
        Path to the source .h5ad file.
    output_path : str
        Where to write the demo subset.
    n_spots : int
        Number of spots to keep (default 200).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    adata : AnnData
    """
    import scanpy as sc
    import scipy.sparse

    print(f"Reading: {input_path}")
    adata = sc.read_h5ad(input_path)

    total = adata.shape[0]
    if n_spots >= total:
        indices = np.arange(total)
    else:
        rng = np.random.RandomState(random_seed)
        indices = np.sort(rng.choice(total, size=n_spots, replace=False))

    adata_demo = adata[indices].copy()

    if "spatial_img_crops" in adata_demo.obsm:
        crops = adata_demo.obsm["spatial_img_crops"]
        if scipy.sparse.issparse(crops):
            crops = crops.toarray()
        adata_demo.obsm["spatial_img_crops"] = crops

    if "protein_names" in adata_demo.uns:
        adata_demo.uns["protein_names"] = np.array(
            list(adata_demo.uns["protein_names"]), dtype=str
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    adata_demo.write(output_path)
    print(f"Demo data saved: {output_path}  ({adata_demo.shape[0]} spots)")
    return adata_demo
