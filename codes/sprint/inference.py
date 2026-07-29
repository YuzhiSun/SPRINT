"""
Inference engine for SPRINT models.

Core workflow
-------------
1. Load dataset from .h5ad via sprint.data
2. Instantiate model, load trained weights
3. Run forward pass on test split → CSV predictions
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ===========================================================================
# Weight loading
# ===========================================================================

def load_weights(model, weight_path, device="cpu"):
    """Load model weights from a PyTorch checkpoint file (state_dict format).

    Supports checkpoints saved as:
      - bare state_dict
      - dict with key "model_state_dict", "model", or "state_dict"
    """
    weight_path = Path(weight_path)
    checkpoint = torch.load(str(weight_path), map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    model.load_state_dict(checkpoint, strict=False)
    return model


# ===========================================================================
# Single-model inference
# ===========================================================================

def predict(model, data_loader, device="cpu"):
    """Run inference on a DataLoader and return predictions + targets as numpy arrays.

    Returns
    -------
    preds : ndarray  (N_spots, N_proteins)
    targets : ndarray  (N_spots, N_proteins)
    """
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for img, rna, label in data_loader:
            img, rna = img.to(device), rna.to(device)
            out = model(img, rna)
            if out.dim() == 3:
                out = out.squeeze(1)
            preds_list.append(out.detach().cpu().numpy())
            targets_list.append(label.numpy())

    preds = np.vstack(preds_list)
    targets = np.vstack(targets_list)
    return preds, targets


def predict_to_csv(
    model,
    data_loader,
    test_indices,
    protein_names,
    output_dir,
    model_label,
    device="cpu",
    save_targets=True,
):
    """Run inference and save predictions (and optionally targets) to CSV.

    Parameters
    ----------
    model : nn.Module
    data_loader : DataLoader
        Test-set loader.
    test_indices : ndarray
        Flat array of test spot indices.
    protein_names : list[str]
        Ordered protein column names.
    output_dir : str or Path
    model_label : str
        Prefix for output CSV filename.
    device : str
    save_targets : bool

    Returns
    -------
    pred_df : DataFrame
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preds, targets = predict(model, data_loader, device)

    # Predictions
    pred_df = pd.DataFrame(preds, columns=protein_names)
    pred_df.insert(0, "test_index", np.asarray(test_indices))
    pred_path = output_dir / f"{model_label}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(
        f"[{model_label}] Predictions saved → {pred_path}  "
        f"({len(pred_df)} spots × {len(protein_names)} proteins)"
    )

    # Targets
    if save_targets:
        target_df = pd.DataFrame(targets, columns=protein_names)
        target_df.insert(0, "test_index", np.asarray(test_indices))
        tgt_path = output_dir / f"{model_label}_targets.csv"
        target_df.to_csv(tgt_path, index=False)

    return pred_df


# ===========================================================================
# Batch export (multiple models on same data)
# ===========================================================================

def export_multiple_models(
    model_specs,
    weights_root,
    output_dir,
    data_loader,
    test_indices,
    protein_names,
    num_genes,
    num_proteins,
    device="cpu",
):
    """Export predictions for multiple models over the same dataset.

    Parameters
    ----------
    model_specs : list[dict]
        Each spec: {"label": str, "model_cls": class, "weight_dir": str}
        weight_dir is relative to weights_root.
    weights_root : Path
    output_dir : Path
    data_loader : DataLoader
    test_indices : ndarray
    protein_names : list[str]
    num_genes, num_proteins : int
    device : str
    """
    import gc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_root = Path(weights_root)
    summary_rows = []

    for i, spec in enumerate(model_specs):
        label = spec["label"]
        model_cls = spec["model_cls"]
        weight_dir = weights_root / spec["weight_dir"]
        weight_path = weight_dir / "best_model.pth"

        if not weight_path.exists():
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        print(f"\n[{i + 1}/{len(model_specs)}] {label}  ({model_cls.__name__})")
        print(f"  Weights: {weight_path}")

        model = model_cls(num_proteins=num_proteins, num_genes=num_genes).to(device)
        load_weights(model, weight_path, device)

        preds, targets = predict(model, data_loader, device)

        pred_df = pd.DataFrame(preds, columns=protein_names)
        pred_df.insert(0, "test_index", np.asarray(test_indices))
        pred_path = output_dir / f"{label}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        target_df = pd.DataFrame(targets, columns=protein_names)
        target_df.insert(0, "test_index", np.asarray(test_indices))
        tgt_path = output_dir / f"{label}_targets.csv"
        target_df.to_csv(tgt_path, index=False)

        summary_rows.append(
            {
                "Model": label,
                "Class": model_cls.__name__,
                "Weights": str(weight_path),
                "Predictions": str(pred_path),
                "Targets": str(tgt_path),
                "Rows": len(pred_df),
                "Proteins": len(protein_names),
            }
        )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(summary_rows)
