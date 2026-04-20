from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def infer_subject_and_class(data_dir: str | Path) -> tuple[Optional[int], Optional[int]]:
    data_path = Path(data_dir).resolve()
    subject_no = None
    num_classes = None

    for part in reversed(data_path.parts):
        if subject_no is None:
            match = re.search(r"Subject[\s_-]*(\d+)", part, re.IGNORECASE)
            if match:
                subject_no = int(match.group(1))

        if num_classes is None:
            match = re.search(r"(\d+)_class", part, re.IGNORECASE)
            if match:
                num_classes = int(match.group(1))

        if subject_no is not None and num_classes is not None:
            break

    return subject_no, num_classes


def build_run_paths(
    project_root: str | Path,
    model_name: str,
    data_dir: str | Path,
    num_classes: Optional[int] = None,
    subject_no: Optional[int] = None,
) -> dict[str, Path | Optional[int]]:
    detected_subject, detected_classes = infer_subject_and_class(data_dir)
    subject_value = subject_no if subject_no is not None else detected_subject
    class_value = num_classes if num_classes is not None else detected_classes

    subject_folder = f"Subject_{subject_value:02d}" if subject_value is not None else "Subject_unknown"
    class_folder = f"{class_value}_class" if class_value is not None else "classes_unknown"
    run_stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    run_dir = Path(project_root) / "results" / subject_folder / class_folder / model_name / run_stamp
    figures_dir = run_dir / "figures"
    metrics_dir = run_dir / "metrics"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"

    for path in (figures_dir, metrics_dir, checkpoints_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "figures_dir": figures_dir,
        "metrics_dir": metrics_dir,
        "checkpoints_dir": checkpoints_dir,
        "logs_dir": logs_dir,
        "subject_no": subject_value,
        "num_classes": class_value,
    }


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def save_classification_outputs(
    metrics_dir: str | Path,
    y_true: Iterable[int],
    y_pred: Iterable[int],
    class_labels: list[str],
    prefix: str,
) -> dict:
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    y_true = list(y_true)
    y_pred = list(y_pred)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_labels,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    (metrics_dir / f"{prefix}_classification_report.txt").write_text(report_text, encoding="utf-8")
    save_json(metrics_dir / f"{prefix}_classification_report.json", report_dict)
    save_json(
        metrics_dir / f"{prefix}_summary.json",
        {
            "accuracy": float(report_dict["accuracy"]),
            "macro_avg": report_dict.get("macro avg", {}),
            "weighted_avg": report_dict.get("weighted avg", {}),
            "num_samples": len(y_true),
            "num_classes": len(class_labels),
        },
    )

    np.save(metrics_dir / f"{prefix}_confusion_matrix.npy", cm)

    return {
        "report_dict": report_dict,
        "report_text": report_text,
        "confusion_matrix": cm,
        "report_json_path": metrics_dir / f"{prefix}_classification_report.json",
        "report_text_path": metrics_dir / f"{prefix}_classification_report.txt",
        "summary_path": metrics_dir / f"{prefix}_summary.json",
        "confusion_matrix_path": metrics_dir / f"{prefix}_confusion_matrix.npy",
    }


def save_confusion_matrix_figure(
    save_path: str | Path,
    cm: np.ndarray,
    class_labels: list[str],
    title: str,
    dpi: int = 300,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    tick_marks = np.arange(len(class_labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                int(cm[row, col]),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
