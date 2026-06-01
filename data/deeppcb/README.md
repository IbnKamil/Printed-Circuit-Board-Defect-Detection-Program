# DeepPCB dataset

This folder contains the official DeepPCB dataset copied from:

```text
https://github.com/tangsanli5201/DeepPCB
```

The dataset is provided by the upstream authors for research purposes. See:

```text
data/deeppcb/LICENSE.DeepPCB
```

Main data folder:

```text
data/deeppcb/PCBData/
```

DeepPCB contains aligned `*_temp.jpg` and `*_test.jpg` image pairs plus `*.txt`
annotations. In this project:

- pair-based CLI/evaluation uses the original DeepPCB reference-based structure;
- Streamlit single-image demo treats `*_temp.jpg` as `normal` and `*_test.jpg`
  as `defective`.
