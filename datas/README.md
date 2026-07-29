# SPRINT data and pretrained models

The datasets and pretrained checkpoints required by the SPRINT notebooks are too large to be hosted directly in the GitHub repository. They will be distributed separately through Zenodo.

## Download

> **Zenodo:** [SPRINT datasets and pretrained models](https://doi.org/10.5281/zenodo.XXXXXXX)

The DOI above is a placeholder and should be replaced after the Zenodo record is published.

Download the complete archive and extract it into this `datas/` directory without changing the internal folder names.

## Expected layout

```text
datas/
├── brain/
│   └── Brain_Multimodal_Final_256px.h5ad
├── breast/
│   ├── Breast_Human_Final.h5ad
│   └── Breast_Mouse_Final.h5ad
├── msi/
│   ├── mouse_brain_1/
│   │   └── mouse_brain_1_Processed.h5ad
│   └── mouse_brain_2/
│       └── mouse_brain_2_Processed.h5ad
├── spleen/
│   ├── mouse_spleen_1/
│   │   └── mouse_spleen_1_Processed.h5ad
│   └── mouse_spleen_2/
│       └── mouse_spleen_2_Processed.h5ad
├── tonsil/
│   ├── Tonsil_1_Final.h5ad
│   └── Tonsil_2_Final.h5ad
├── models/
│   ├── brain/A2/best_model.pth
│   ├── breast/C2/best_model.pth
│   ├── msi_mouse_brain/A2/best_model.pth
│   ├── spleen/C2/best_model.pth
│   └── tonsil/
│       ├── A2/best_model.pth
│       ├── HE_Only/best_model.pth
│       └── RNA_Only/best_model.pth
└── outputs/
    ├── brain/
    ├── breast/
    ├── msi_mouse_brain/
    ├── spleen/
    └── tonsil/
```

The `outputs/` directories may initially be empty. They are populated by the corresponding `produce_*_predictions.ipynb` notebooks.

## Integrity and licensing

Checksums will be supplied with the Zenodo archive. Verify the downloaded files before running the notebooks and retain the filenames shown above.

The archive contains processed versions of publicly available datasets. Users remain responsible for complying with the terms and licenses of the original data providers. Dataset provenance and source links are documented in the accompanying manuscript.
