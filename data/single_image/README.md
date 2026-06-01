# Single-image dataset

Place the local dataset for Streamlit auto-training here:

```text
data/single_image/
├── normal/
│   ├── image_001.jpg
│   └── ...
└── defective/
    ├── image_101.jpg
    └── ...
```

When `app/streamlit_app.py` starts, it checks for:

```text
outputs/single_image_model.pt
```

If the checkpoint exists, the app loads it and prints metrics/dataset counts in
the PyCharm console. If it does not exist, the app trains a model from this
folder and then saves the checkpoint.

You can use another local dataset path by setting:

```powershell
$env:PCB_SINGLE_IMAGE_DATASET_DIR="C:\path\to\dataset"
```
