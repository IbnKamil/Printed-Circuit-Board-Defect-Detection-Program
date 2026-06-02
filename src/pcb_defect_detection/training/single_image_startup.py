"""Startup report formatting for Streamlit."""

from __future__ import annotations

from pathlib import Path
from typing import Any


_SPLIT_NAMES_RU = {
    "total": "всего",
    "train": "обучение",
    "validation": "валидация",
    "test": "тест",
}


def format_single_image_startup_report(
    dataset_root: Path,
    primary_metrics: dict[str, Any] | None = None,
) -> str:
    """Format primary DeepPCB-aware metrics as a PyCharm console report."""

    primary_metrics = primary_metrics or {}
    dataset = primary_metrics.get("dataset", {}) or {}
    lines = [
        "",
        "========== ОТЧЕТ ЗАПУСКА PCB STREAMLIT =========",
        "ОСНОВНОЙ МЕТОД",
        "  Режим      : загрузка одного изображения с DeepPCB-aware анализом",
        "  *_temp.jpg : NORMAL, эталонное изображение считается исправной платой",
        "  *_test.jpg : сравнение с найденным скрытым эталоном *_temp.jpg",
        "  Примечание : это единственный метод, используемый Streamlit-интерфейсом.",
        "",
        "МЕТРИКИ ОСНОВНОГО МЕТОДА",
        _format_primary_metrics_table(primary_metrics),
        "",
        "ДАТАСЕТ",
        f"  Корневая папка DeepPCB : {dataset_root}",
        _format_dataset_counts(dataset),
        "",
        "ПРИМЕЧАНИЯ",
        "  Метрики выше оценивают фактическое поведение интерфейса на validation/test split.",
        "================================================",
    ]
    return "\n".join(lines)


def _format_primary_metrics_table(primary_metrics: dict[str, Any]) -> str:
    if not primary_metrics:
        return "  Метрики    : еще не рассчитаны"

    lines = ["  Выборка      Объектов   Accuracy  Precision     Recall         F1     TN     FP     FN     TP"]
    for split_name in ("validation", "test"):
        values = primary_metrics.get(split_name, {})
        if not values:
            continue
        display_name = _SPLIT_NAMES_RU.get(split_name, split_name)
        lines.append(
            "  "
            f"{display_name:<10} "
            f"{int(values.get('samples', 0)):>8} "
            f"{_fmt(values.get('accuracy')):>10} "
            f"{_fmt(values.get('precision')):>10} "
            f"{_fmt(values.get('recall')):>10} "
            f"{_fmt(values.get('f1')):>10} "
            f"{int(values.get('true_negative', 0)):>6} "
            f"{int(values.get('false_positive', 0)):>6} "
            f"{int(values.get('false_negative', 0)):>6} "
            f"{int(values.get('true_positive', 0)):>6}"
        )
    lines.extend(
        [
            "  Эти метрики оценивают реальный сценарий Streamlit для DeepPCB:",
            "  *_temp -> normal, *_test -> скрытое сравнение с template.",
        ]
    )
    return "\n".join(lines)


def _format_dataset_counts(dataset: dict[str, Any]) -> str:
    if not dataset:
        return "  Количество : данные о split не найдены в кэше метрик"

    rows = [
        ("total", dataset.get("total"), dataset.get("total_by_class", {})),
        ("train", dataset.get("train"), dataset.get("train_by_class", {})),
        ("validation", dataset.get("validation"), dataset.get("validation_by_class", {})),
        ("test", dataset.get("test"), dataset.get("test_by_class", {})),
    ]
    lines = ["  Выборка      Всего    Normal    Defective"]
    for name, total, by_class in rows:
        if total is None:
            continue
        normal = by_class.get("normal", "-") if isinstance(by_class, dict) else "-"
        defective = by_class.get("defective", "-") if isinstance(by_class, dict) else "-"
        display_name = _SPLIT_NAMES_RU.get(name, name)
        lines.append(f"  {display_name:<10} {int(total):>6} {str(normal):>9} {str(defective):>12}")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
