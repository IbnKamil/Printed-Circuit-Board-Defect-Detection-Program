"""Streamlit demo UI for pair-based PCB defect detection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_defect_detection.detection.classical import ClassicalDetectorConfig
from pcb_defect_detection.inference.pipeline import PCBDefectPipeline
from pcb_defect_detection.visualization import difference_heatmap, draw_detections, mask_to_rgb


st.set_page_config(page_title="PCB Defect Detection", layout="wide")
st.title("PCB Defect Detection on Template/Tested Pairs")
st.caption("Reference-based DeepPCB-style comparison with bbox visualization.")

with st.sidebar:
    st.header("Parameters")
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
    diff_threshold = st.slider("Difference threshold", 1, 80, 22, 1)
    checkpoint_file = st.file_uploader("Optional PatchCNN checkpoint (.pt)", type=["pt"])

template_file = st.file_uploader("Template image", type=["png", "jpg", "jpeg", "bmp"])
tested_file = st.file_uploader("Tested image", type=["png", "jpg", "jpeg", "bmp"])

if template_file and tested_file:
    template = np.asarray(Image.open(template_file).convert("RGB"))
    tested = np.asarray(Image.open(tested_file).convert("RGB"))

    detector_config = ClassicalDetectorConfig(
        confidence_threshold=confidence_threshold,
        diff_threshold=diff_threshold,
    )

    checkpoint_path = None
    if checkpoint_file is not None:
        checkpoint_path = ROOT / "outputs" / "uploaded_checkpoint.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(checkpoint_file.getvalue())

    try:
        pipeline = PCBDefectPipeline(
            detector_config=detector_config,
            checkpoint_path=checkpoint_path,
        )
        result, artifacts = pipeline.predict_arrays(template, tested)
        overlay = draw_detections(artifacts["tested_rgb"], result.detections)
        heatmap = difference_heatmap(artifacts["abs_diff"])
        mask = mask_to_rgb(artifacts["mask"])

        col1, col2, col3 = st.columns(3)
        col1.image(artifacts["template_rgb"], caption="Template", use_container_width=True)
        col2.image(artifacts["tested_rgb"], caption="Tested", use_container_width=True)
        col3.image(overlay, caption="Detected defects", use_container_width=True)

        col4, col5 = st.columns(2)
        col4.image(mask, caption="Binary difference mask", use_container_width=True)
        col5.image(heatmap, caption="Difference heatmap", use_container_width=True)

        st.subheader(result.conclusion)
        if result.detections:
            st.dataframe([detection.to_dict() for detection in result.detections])
        else:
            st.info("No defect proposals passed the confidence threshold.")

        report_json = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        st.download_button(
            "Download JSON report",
            data=report_json,
            file_name="pcb_defect_report.json",
            mime="application/json",
        )
    except Exception as exc:  # noqa: BLE001 - Streamlit should show readable UI error.
        st.error(str(exc))
else:
    st.info("Upload both template and tested images to start analysis.")
