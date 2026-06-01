# Optional single-image dataset

The repository now includes DeepPCB in:

```text
data/deeppcb/PCBData/
```

Streamlit uses that folder by default. For the single-image demo, images named
`*_temp.jpg` are treated as `normal`, and images named `*_test.jpg` are treated
as `defective`.

You only need this `data/single_image/` folder if you want to replace DeepPCB
with your own binary dataset:

```text
data/single_image/
├── normal/
│   ├── image_001.jpg
│   └── ...
└── defective/
    ├── image_101.jpg
    └── ...
```

To use a custom dataset without moving files, set:

```powershell
$env:PCB_SINGLE_IMAGE_DATASET_DIR="C:\path\to\dataset"
```
