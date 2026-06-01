"""Streamlit demo UI for PCB defect detection."""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_defect_detection.detection.classical import ClassicalDetectorConfig
from pcb_defect_detection.inference.pipeline import PCBDefectPipeline
from pcb_defect_detection.models.single_image_cnn import predict_single_image
from pcb_defect_detection.training.train_single_image_classifier import train_single_image_classifier
from pcb_defect_detection.visualization import difference_heatmap, draw_detections, mask_to_rgb

DEFAULT_SINGLE_MODEL_PATH = ROOT / "outputs" / "single_image_model.pt"
UPLOADED_DATASET_DIR = ROOT / "outputs" / "streamlit_uploaded_dataset"


def _render_status_square(prediction: dict[str, float | str | bool]) -> None:
    is_defective = bool(prediction["is_defective"])
    color = "#d32f2f" if is_defective else "#2e7d32"
    label = "DEFECT" if is_defective else "NORMAL"
    description = "Плата с дефектом" if is_defective else "Плата в нормальном состоянии"
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:18px;margin:12px 0;">
          <div style="width:120px;height:120px;background:{color};border-radius:8px;
                      border:4px solid #111;display:flex;align-items:center;
                      justify-content:center;color:white;font-weight:800;font-size:18px;">
            {label}
          </div>
          <div>
            <h3 style="margin-bottom:4px;">{description}</h3>
            <p style="margin-top:0;">Красный квадрат означает дефект, зелёный — норму.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _extract_uploaded_zip(uploaded_file, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    zip_path = destination.parent / "streamlit_dataset.zip"
    zip_path.write_bytes(uploaded_file.getvalue())
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not str(target).startswith(str(destination.resolve())):
                raise ValueError("Unsafe path detected in ZIP archive.")
        archive.extractall(destination)


st.set_page_config(page_title="PCB Defect Detection", layout="wide")
st.title("PCB Defect Detection")
st.caption(
    "Single-photo defect check for demos plus DeepPCB template/tested pair comparison."
)

single_tab, pair_tab = st.tabs(
    ["Single photo inspection", "DeepPCB template/tested pair"]
)

with single_tab:
    st.header("Проверка одной фотографии платы")
    st.write(
        "Загрузите одно изображение PCB. Если бинарная модель обучена, приложение "
        "выдаст заключение: нормальная плата или плата с дефектом."
    )

    model_path = DEFAULT_SINGLE_MODEL_PATH
    uploaded_checkpoint = st.file_uploader(
        "Optional: загрузить обученный single-image checkpoint (.pt)",
        type=["pt"],
        key="single_checkpoint",
    )
    if uploaded_checkpoint is not None:
        model_path = ROOT / "outputs" / "uploaded_single_image_model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(uploaded_checkpoint.getvalue())

    model_exists = model_path.exists()
    image_file = st.file_uploader(
        "Фотография платы для проверки",
        type=["png", "jpg", "jpeg", "bmp"],
        key="single_image",
    )

    if image_file is not None:
        image = Image.open(image_file).convert("RGB")
        image_rgb = np.asarray(image)
        left, right = st.columns([2, 1])
        left.image(image_rgb, caption="Uploaded PCB image", use_container_width=True)

        if model_exists:
            try:
                prediction = predict_single_image(image_rgb, model_path)
                _render_status_square(prediction)
                right.metric("Confidence", f"{prediction['confidence']:.2%}")
                right.metric(
                    "Defect probability",
                    f"{prediction['defective_probability']:.2%}",
                )
                right.json(prediction)
            except Exception as exc:  # noqa: BLE001 - Streamlit should show readable UI error.
                right.error(str(exc))
        else:
            right.warning(
                "Модель для одиночной фотографии ещё не обучена. "
                "Загрузите dataset ниже и нажмите Train model."
            )

    st.divider()
    st.subheader("Обучение модели, если checkpoint отсутствует")
    st.write(
        "Загрузите ZIP-архив с двумя папками классов. Минимальная структура:"
    )
    st.code(
        "dataset.zip\n"
        "├── normal/\n"
        "│   ├── image_001.jpg\n"
        "│   └── ...\n"
        "└── defective/\n"
        "    ├── image_101.jpg\n"
        "    └── ...",
        language="text",
    )
    st.info(
        "Для реальной точности нужны реальные размеченные изображения. "
        "Для защиты можно обучить маленькую demo-модель на небольшом наборе, "
        "но качество будет зависеть от данных."
    )

    dataset_zip = st.file_uploader(
        "Dataset ZIP для обучения normal/defective",
        type=["zip"],
        key="single_dataset_zip",
    )
    train_col1, train_col2, train_col3 = st.columns(3)
    epochs = train_col1.number_input("Epochs", min_value=1, max_value=50, value=5)
    batch_size = train_col2.number_input("Batch size", min_value=1, max_value=64, value=8)
    image_size = train_col3.selectbox("Image size", options=[96, 128, 160, 224], index=1)

    if st.button("Train single-image model", disabled=dataset_zip is None):
        try:
            if dataset_zip is None:
                st.error("Upload dataset ZIP first.")
            else:
                _extract_uploaded_zip(dataset_zip, UPLOADED_DATASET_DIR)
                with st.spinner("Training single-image classifier..."):
                    metrics = train_single_image_classifier(
                        dataset_root=UPLOADED_DATASET_DIR,
                        output_path=DEFAULT_SINGLE_MODEL_PATH,
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        image_size=int(image_size),
                    )
                st.success(f"Model saved to {DEFAULT_SINGLE_MODEL_PATH}")
                st.json(metrics)
        except Exception as exc:  # noqa: BLE001 - Streamlit should show readable UI error.
            st.error(str(exc))

with pair_tab:
    st.header("DeepPCB: сравнение template + tested")
    st.write(
        "Этот режим соответствует DeepPCB: загружаются эталонное изображение "
        "и проверяемое изображение одинакового размера."
    )

    with st.sidebar:
        st.header("DeepPCB pair parameters")
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
        diff_threshold = st.slider("Difference threshold", 1, 80, 22, 1)
        checkpoint_file = st.file_uploader(
            "Optional PatchCNN checkpoint (.pt)",
            type=["pt"],
            key="pair_checkpoint",
        )

    template_file = st.file_uploader(
        "Template image",
        type=["png", "jpg", "jpeg", "bmp"],
        key="template_image",
    )
    tested_file = st.file_uploader(
        "Tested image",
        type=["png", "jpg", "jpeg", "bmp"],
        key="tested_image",
    )

    if template_file and tested_file:
        template = np.asarray(Image.open(template_file).convert("RGB"))
        tested = np.asarray(Image.open(tested_file).convert("RGB"))

        detector_config = ClassicalDetectorConfig(
            confidence_threshold=confidence_threshold,
            diff_threshold=diff_threshold,
        )

        checkpoint_path = None
        if checkpoint_file is not None:
            checkpoint_path = ROOT / "outputs" / "uploaded_pair_checkpoint.pt"
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
        st.info("Upload both template and tested images to start pair-based analysis.")

