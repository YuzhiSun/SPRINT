"""Prediction export utilities for SPRINT inference notebooks.

Each produce_*_predictions.ipynb uses ``export_model_specs`` (or
``export_msi_model_specs`` for MSI data) to load trained weights and
save predictions + targets as CSV files.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# GPU selection (must run BEFORE torch import)
# -------------------------------------------------------------------------

def select_least_used_cuda_before_torch_import():
    """Pick the least-utilised GPU via nvidia-smi and set CUDA_VISIBLE_DEVICES."""
    current_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    allowed_physical_ids = None
    if current_visible and current_visible.strip() not in {"", "NoDevFiles"}:
        try:
            allowed_physical_ids = [
                int(item.strip())
                for item in current_visible.split(",")
                if item.strip()
            ]
        except ValueError:
            allowed_physical_ids = None

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
    except Exception as exc:
        print(
            f"nvidia-smi GPU selection failed before torch import: {exc}. "
            "Keeping default CUDA visibility."
        )
        return

    gpu_rows = []
    for line in result.stdout.strip().splitlines():
        index, used, total = [item.strip() for item in line.split(",")]
        physical_index = int(index)
        if allowed_physical_ids is not None and physical_index not in allowed_physical_ids:
            continue
        gpu_rows.append(
            {
                "index": physical_index,
                "used_mb": int(used),
                "total_mb": int(total),
                "used_ratio": int(used) / max(int(total), 1),
            }
        )

    if not gpu_rows:
        print(
            "No visible CUDA devices found by nvidia-smi. "
            "Keeping default CUDA visibility."
        )
        return

    best = min(gpu_rows, key=lambda row: (row["used_mb"], row["used_ratio"], row["index"]))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best["index"])
    print("Detected CUDA devices before torch import:")
    for row in gpu_rows:
        marker = " <-- selected as cuda:0" if row["index"] == best["index"] else ""
        print(
            f"  physical={row['index']} used={row['used_mb']} MB / {row['total_mb']} MB "
            f"({row['used_ratio']:.1%}){marker}"
        )


# -------------------------------------------------------------------------
# Project root detection
# -------------------------------------------------------------------------

def find_project_root(start):
    """Locate the SPRINT project root by walking up from *start*.

    Looks for a directory containing ``codes/sprint/__init__.py``.
    """
    start = Path(start).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "codes" / "sprint" / "__init__.py").exists():
            return candidate
    raise RuntimeError(f"Cannot find project root (codes/sprint/__init__.py) from {start}")


# -------------------------------------------------------------------------
# Weight discovery
# -------------------------------------------------------------------------

def find_model_weights(model_spec, model_save_root):
    """Search *model_save_root* for ``best_model.pth`` among candidate dirs."""
    model_save_root = Path(model_save_root)
    tried = []
    for candidate in model_spec["model_dir_candidates"]:
        candidate_path = model_save_root / candidate / "best_model.pth"
        tried.append(candidate_path)
        if candidate_path.exists():
            return candidate_path, candidate
    tried_text = "\n".join(str(path) for path in tried)
    raise FileNotFoundError(
        f"No trained weights found for {model_spec['label']}. Tried:\n{tried_text}"
    )


# -------------------------------------------------------------------------
# Prediction helpers
# -------------------------------------------------------------------------

def predict_with_model(model, loader, device):
    """Run forward pass and return (preds, targets) as numpy arrays."""
    import torch

    preds_list = []
    targets_list = []
    model.eval()
    with torch.no_grad():
        for img, rna, label in loader:
            img = img.to(device)
            rna = rna.to(device)
            out = model(img, rna)
            if out.dim() == 3:
                out = out.squeeze(1)
            preds_list.append(out.cpu().numpy())
            targets_list.append(label.numpy())
    return np.vstack(preds_list), np.vstack(targets_list)


def make_result_df(values, feature_names):
    """Build a DataFrame with test_index column + feature columns."""
    df = pd.DataFrame(values, columns=feature_names)
    df.insert(0, "test_index", np.arange(len(df), dtype=int))
    return df


# -------------------------------------------------------------------------
# Batch export (protein datasets)
# -------------------------------------------------------------------------

def export_model_specs(
    model_specs,
    model_save_root,
    output_dir,
    loader,
    device,
    num_targets,
    num_genes,
    target_names,
    load_weights_fn,
):
    """Iterate *model_specs*, load weights, predict, and save CSV results."""
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    reference_targets = None
    last_pred_df = None
    last_target_df = None

    for spec in model_specs:
        weights_path, model_dir_name = find_model_weights(spec, model_save_root)
        model = spec["model_class"](num_proteins=num_targets, num_genes=num_genes).to(device)
        load_weights_fn(model, str(weights_path), device)

        preds, targets = predict_with_model(model, loader, device)
        if reference_targets is None:
            reference_targets = targets
        elif not np.array_equal(reference_targets, targets):
            raise ValueError(
                f"Targets changed while exporting {spec['label']}; check the dataloader order."
            )

        pred_df = make_result_df(preds, target_names)
        target_df = make_result_df(targets, target_names)

        pred_path = output_dir / f"{spec['output_prefix']}_predictions.csv"
        target_path = output_dir / f"{spec['output_prefix']}_targets.csv"
        pred_df.to_csv(pred_path, index=False)
        target_df.to_csv(target_path, index=False)

        saved_files.append(
            {
                "Model": spec["label"],
                "Weights": str(weights_path),
                "ModelDir": model_dir_name,
                "Predictions": str(pred_path),
                "Targets": str(target_path),
                "Rows": len(pred_df),
                "TargetsCount": len(target_names),
            }
        )
        print(f"[{spec['label']}] saved predictions: {pred_path}")
        print(f"[{spec['label']}] saved targets:     {target_path}")

        last_pred_df = pred_df
        last_target_df = target_df
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(saved_files), last_pred_df, last_target_df


# -------------------------------------------------------------------------
# MSI-specific helpers
# -------------------------------------------------------------------------

def load_msi_weights(model, weights_path, device):
    """Load MSI model weights (supports {"model_state_dict": ...} wrapper)."""
    import torch

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return checkpoint


def predict_msi_model(model, loader, device, target_mean, target_std):
    """Run MSI forward pass and de-normalise (z-score → original scale)."""
    import torch

    preds_z = []
    targets_z = []
    model.eval()
    with torch.no_grad():
        for img, rna, y in loader:
            out = model(img.to(device), rna.to(device))
            preds_z.append(out.cpu().numpy())
            targets_z.append(y.numpy())
    preds_z = np.vstack(preds_z)
    targets_z = np.vstack(targets_z)
    preds = preds_z * target_std + target_mean
    targets = targets_z * target_std + target_mean
    return preds, targets


def export_msi_model_specs(
    model_specs,
    model_save_root,
    output_dir,
    loader,
    device,
    model_class,
    num_genes,
    num_metabolites,
    target_names,
    target_mean,
    target_std,
):
    """Iterate MSI *model_specs*, load weights, predict, save CSV results."""
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    reference_targets = None
    last_pred_df = None
    last_target_df = None

    for spec in model_specs:
        weights_path, model_dir_name = find_model_weights(spec, model_save_root)
        model = model_class(
            spec["scheme"],
            num_genes,
            num_metabolites,
            freeze_image=spec.get("freeze_image", False),
        ).to(device)
        checkpoint = load_msi_weights(model, weights_path, device)
        mean = (
            np.asarray(checkpoint.get("target_mean", target_mean), dtype=np.float32)
            if isinstance(checkpoint, dict)
            else target_mean
        )
        std = (
            np.asarray(checkpoint.get("target_std", target_std), dtype=np.float32)
            if isinstance(checkpoint, dict)
            else target_std
        )

        preds, targets = predict_msi_model(model, loader, device, mean, std)
        if reference_targets is None:
            reference_targets = targets
        elif not np.array_equal(reference_targets, targets):
            raise ValueError(
                f"Targets changed while exporting {spec['label']}; "
                "check the dataloader order or target scaling."
            )

        pred_df = make_result_df(preds, target_names)
        target_df = make_result_df(targets, target_names)

        pred_path = output_dir / f"{spec['output_prefix']}_predictions.csv"
        target_path = output_dir / f"{spec['output_prefix']}_targets.csv"
        pred_df.to_csv(pred_path, index=False)
        target_df.to_csv(target_path, index=False)

        saved_files.append(
            {
                "Model": spec["label"],
                "Weights": str(weights_path),
                "ModelDir": model_dir_name,
                "Predictions": str(pred_path),
                "Targets": str(target_path),
                "Rows": len(pred_df),
                "TargetsCount": len(target_names),
            }
        )
        print(f"[{spec['label']}] saved predictions: {pred_path}")
        print(f"[{spec['label']}] saved targets:     {target_path}")

        last_pred_df = pred_df
        last_target_df = target_df
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(saved_files), last_pred_df, last_target_df
