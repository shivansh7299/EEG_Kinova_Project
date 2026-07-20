"""
Build Table I (offline test accuracy) from latest run per subject per model.
Usage: python paper/build_table_i.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

SUBJECTS = ["01", "02", "03", "04", "10", "11", "12", "13", "14", "15", "16", "17"]
MODELS = {
    "EEGNet": "eegnet_new_summary.json",
    "CTNet": "ctnet_summary.json",
    "FBMSNet": "fbmsnet_summary.json",
}


def run_timestamp(path: Path) -> str:
    match = re.search(r"run_(\d{8}_\d{6})", str(path))
    return match.group(1) if match else ""


def latest_summary(subject: str, model: str, filename: str) -> dict | None:
    pattern = f"Subject_{subject}/2_class/{model}/run_*/metrics/{filename}"
    paths = list(RESULTS_DIR.glob(pattern))
    if not paths:
        return None
    latest = max(paths, key=run_timestamp)
    data = json.loads(latest.read_text(encoding="utf-8"))
    return {
        "accuracy_pct": round(data["accuracy"] * 100, 1),
        "num_samples": data.get("num_samples"),
        "run_id": latest.parent.parent.name,
        "path": str(latest.relative_to(PROJECT_ROOT)),
    }


def build_table() -> dict:
    table: dict[str, dict[str, dict | None]] = {}
    for subject in SUBJECTS:
        table[subject] = {}
        for model, fname in MODELS.items():
            table[subject][model] = latest_summary(subject, model, fname)
    return table


def format_cell(entry: dict | None) -> str:
    if entry is None:
        return "---"
    return f"{entry['accuracy_pct']:.1f}"


def mean_accuracy(table: dict, model: str) -> float | None:
    values = [
        table[s][model]["accuracy_pct"]
        for s in SUBJECTS
        if table[s][model] is not None
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 1)


def print_latex_rows(table: dict) -> None:
    print("% --- Paste into ieee_bsn2026.tex Table I ---")
    for subject in SUBJECTS:
        label = f"S{int(subject)}"
        cells = [label] + [format_cell(table[subject][m]) for m in MODELS]
        print(" & ".join(cells) + r" \\")
    mean_cells = ["Mean"] + [
        f"{mean_accuracy(table, m):.1f}" if mean_accuracy(table, m) is not None else "---"
        for m in MODELS
    ]
    print(" & ".join(mean_cells) + r" \\")


def main() -> None:
    table = build_table()
    out_json = Path(__file__).parent / "table_i_data.json"
    out_json.write_text(json.dumps(table, indent=2), encoding="utf-8")

    print("=" * 72)
    print("TABLE I — Offline test accuracy (%) — latest run per subject/model")
    print("=" * 72)
    header = ["Subject", *MODELS.keys(), "Run IDs (latest)"]
    print(f"{header[0]:<10} {header[1]:>8} {header[2]:>8} {header[3]:>8}")
    print("-" * 72)
    for subject in SUBJECTS:
        row = [f"S{int(subject)}"]
        runs = []
        for model in MODELS:
            entry = table[subject][model]
            row.append(format_cell(entry))
            if entry:
                runs.append(f"{model}:{entry['run_id']}")
        print(f"{row[0]:<10} {row[1]:>8} {row[2]:>8} {row[3]:>8}")
        if runs:
            print(f"           runs: {', '.join(runs)}")
    print("-" * 72)
    for model in MODELS:
        m = mean_accuracy(table, model)
        n = sum(1 for s in SUBJECTS if table[s][model] is not None)
        print(f"Mean {model}: {m:.1f}% (n={n})" if m is not None else f"Mean {model}: ---")
    print()
    print(f"Saved: {out_json.relative_to(PROJECT_ROOT)}")
    print()
    print_latex_rows(table)


if __name__ == "__main__":
    main()
