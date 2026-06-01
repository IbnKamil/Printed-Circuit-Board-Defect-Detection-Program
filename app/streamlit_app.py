"""Streamlit UI for single-photo PCB defect inspection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_defect_detection.inference.single_image import predict_uploaded_pcb_image
from pcb_defect_detection.training.train_single_image_classifier import (
    load_single_image_training_summary,
)

DEFAULT_SINGLE_MODEL_PATH = ROOT / "outputs" / "single_image_model.pt"
DEFAULT_DATASET_ROOT = ROOT / "data" / "deeppcb" / "PCBData"


@st.cache_resource(show_spinner=False)
def _startup_model_check() -> dict:
    """Log model/dataset information once without blocking UI training."""

    print("\n========== PCB STREAMLIT STARTUP ==========")
    print(f"DeepPCB dataset root: {DEFAULT_DATASET_ROOT}")
    print("Primary inference mode: one-upload DeepPCB-aware template lookup")
    print("  *_temp.jpg -> normal")
    print("  *_test.jpg -> hidden comparison with matching *_temp.jpg")
    if DEFAULT_SINGLE_MODEL_PATH.exists():
        summary = load_single_image_training_summary(DEFAULT_SINGLE_MODEL_PATH)
        print("Fallback single-image CNN checkpoint found:")
        print(summary)
        print("===========================================\n")
        return summary
    print("Fallback single-image CNN checkpoint not found; startup training is skipped.")
    print("DeepPCB *_test/*_temp images still work through hidden template comparison.")
    print("===========================================\n")
    return {"checkpoint_path": None}


def _render_status_square(is_defective: bool) -> None:
    color = "#d32f2f" if is_defective else "#2e7d32"
    label = "DEFECT" if is_defective else "NORMAL"
    description = "Плата с дефектом" if is_defective else "Плата в нормальном состоянии"
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:18px;margin:14px 0;">
          <div style="width:128px;height:128px;background:{color};border-radius:10px;
                      border:4px solid #111;display:flex;align-items:center;
                      justify-content:center;color:white;font-weight:800;font-size:18px;">
            {label}
          </div>
          <div>
            <h3 style="margin-bottom:4px;">{description}</h3>
            <p style="margin-top:0;">Зелёный квадрат — норма, красный квадрат — дефект.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="PCB Defect Detection", layout="centered")
st.title("PCB Defect Detection")
st.write("Загрузите одно изображение печатной платы для проверки состояния.")

try:
    _startup_model_check()
except Exception as exc:  # noqa: BLE001 - Streamlit should display readable startup issue.
    print(f"[PCB Streamlit] Model startup failed: {exc}")

uploaded_image = st.file_uploader(
    "Загрузить изображение платы",
    type=["png", "jpg", "jpeg", "bmp"],
)

if uploaded_image is None:
    st.info("Ожидается загрузка одного изображения PCB.")
else:
    image = Image.open(uploaded_image).convert("RGB")
    image_rgb = np.asarray(image)

    try:
        result = predict_uploaded_pcb_image(
            image_rgb=image_rgb,
            filename=uploaded_image.name,
            dataset_root=DEFAULT_DATASET_ROOT,
            checkpoint_path=DEFAULT_SINGLE_MODEL_PATH,
        )
        st.image(result.overlay_rgb, caption="Результат анализа", use_container_width=True)
        _render_status_square(result.is_defective)
        st.write(f"Confidence: **{result.confidence:.2%}**")
        st.caption(result.message)
        if result.detections:
            st.write(f"Найдено областей несоответствия: **{len(result.detections)}**")
            st.dataframe([detection.to_dict() for detection in result.detections[:50]])
    except Exception as exc:  # noqa: BLE001 - Streamlit should display readable prediction issue.
        st.image(image_rgb, caption="Загруженное изображение", use_container_width=True)
        st.error(str(exc))
