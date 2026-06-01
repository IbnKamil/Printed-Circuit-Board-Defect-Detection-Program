"""Streamlit UI for single-photo PCB defect inspection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_defect_detection.models.single_image_cnn import predict_single_image
from pcb_defect_detection.training.single_image_startup import ensure_single_image_model

DEFAULT_SINGLE_MODEL_PATH = ROOT / "outputs" / "single_image_model.pt"
DEFAULT_DATASET_ROOT = ROOT / "data" / "deeppcb" / "PCBData"


@st.cache_resource(show_spinner=False)
def _startup_model_check() -> dict:
    """Check/train the model once per Streamlit process and log summary to console."""

    return ensure_single_image_model(
        checkpoint_path=DEFAULT_SINGLE_MODEL_PATH,
        default_dataset_root=DEFAULT_DATASET_ROOT,
        epochs=3,
        batch_size=32,
        image_size=128,
    )


def _render_status_square(prediction: dict[str, float | str | bool]) -> None:
    is_defective = bool(prediction["is_defective"])
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
st.write("Загрузите фотографию печатной платы для проверки состояния.")

try:
    model_summary = _startup_model_check()
    model_ready = DEFAULT_SINGLE_MODEL_PATH.exists()
except Exception as exc:  # noqa: BLE001 - Streamlit should display readable startup issue.
    model_summary = {"startup_error": str(exc)}
    model_ready = False
    print(f"[PCB Streamlit] Model startup failed: {exc}")

uploaded_image = st.file_uploader(
    "Загрузить изображение платы",
    type=["png", "jpg", "jpeg", "bmp"],
)

if uploaded_image is None:
    st.info("Ожидается загрузка одного изображения PCB.")
elif not model_ready:
    st.error(
        "Модель не найдена и не смогла обучиться автоматически. "
        "Проверьте консоль PyCharm и наличие датасета в data/single_image/normal и "
        "data/single_image/defective."
    )
else:
    image = Image.open(uploaded_image).convert("RGB")
    image_rgb = np.asarray(image)
    st.image(image_rgb, caption="Загруженное изображение", use_container_width=True)

    try:
        prediction = predict_single_image(image_rgb, DEFAULT_SINGLE_MODEL_PATH)
        _render_status_square(prediction)
        st.write(f"Confidence: **{prediction['confidence']:.2%}**")
        st.write(f"Вероятность дефекта: **{prediction['defective_probability']:.2%}**")
    except Exception as exc:  # noqa: BLE001 - Streamlit should display readable prediction issue.
        st.error(str(exc))
