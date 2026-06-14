"""
Generate IEEE BSN paper figures: per-subject EEG-only vs EEG+Vision performance.

HOW TO USE
----------
1. Fill in SUBJECT_ONLINE_DATA below (values from your offline evaluation form).
2. Offline EEG accuracy is auto-loaded from paper/table_i_data.json when present.
3. Run:  python paper/plot_subject_performance.py
4. Figures are saved to:  paper/figures/

Metrics (all in % unless noted):
  - offline_eeg_acc      : offline test accuracy from trained EEGNet (auto)
  - online_eeg_success   : online control success rate, EEG-only mode
  - online_hybrid_success: online control success rate, EEG + Vision mode
  - eeg_vision_agreement : % of time EEG prediction matched OpenCV bar direction
  - command_latency_s    : mean seconds from cue to arm movement (lower is better)
  - usable_trials        : number of valid online trials (for labels only)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# =============================================================================
# >>> EDIT YOUR VALUES HERE (from your offline per-subject form) <<<
# =============================================================================

SUBJECTS = ["S1", "S2", "S3", "S4", "S10"]

SUBJECT_ONLINE_DATA = {
    # Subject: {metric: value}
    "S1": {
        "online_eeg_success": 0.0,       # %  EEG-only online success
        "online_hybrid_success": 0.0,    # %  EEG+Vision online success
        "eeg_vision_agreement": 0.0,     # %  EEG matches OpenCV direction
        "command_latency_s": 0.0,        # seconds (mean)
        "usable_trials": 0,                # count
    },
    "S2": {
        "online_eeg_success": 0.0,
        "online_hybrid_success": 0.0,
        "eeg_vision_agreement": 0.0,
        "command_latency_s": 0.0,
        "usable_trials": 0,
    },
    "S3": {
        "online_eeg_success": 0.0,
        "online_hybrid_success": 0.0,
        "eeg_vision_agreement": 0.0,
        "command_latency_s": 0.0,
        "usable_trials": 0,
    },
    "S4": {
        "online_eeg_success": 0.0,
        "online_hybrid_success": 0.0,
        "eeg_vision_agreement": 0.0,
        "command_latency_s": 0.0,
        "usable_trials": 0,
    },
    "S10": {
        "online_eeg_success": 0.0,
        "online_hybrid_success": 0.0,
        "eeg_vision_agreement": 0.0,
        "command_latency_s": 0.0,
        "usable_trials": 0,
    },
}

# Optional: subjective ratings (1–5) if your form includes them
SUBJECTIVE_RATINGS = {
    # "S1": {"eeg_only": 3.0, "eeg_vision": 4.0},
}

# =============================================================================
# Plot style (IEEE-friendly)
# =============================================================================

PAPER_DIR = Path(__file__).resolve().parent
FIG_DIR = PAPER_DIR / "figures"
TABLE_JSON = PAPER_DIR / "table_i_data.json"

COLOR_OFFLINE = "#4C72B0"
COLOR_EEG = "#DD8452"
COLOR_HYBRID = "#55A868"
COLOR_AGREE = "#8172B2"
COLOR_DELTA_POS = "#55A868"
COLOR_DELTA_NEG = "#C44E52"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_from_csv(csv_path: Path | None = None) -> None:
    """Optional: load online metrics from subject_form_template.csv."""
    import csv

    path = csv_path or PAPER_DIR / "subject_form_template.csv"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            subj = row["subject"].strip()
            if subj not in SUBJECT_ONLINE_DATA:
                continue
            d = SUBJECT_ONLINE_DATA[subj]
            for key in (
                "online_eeg_success", "online_hybrid_success",
                "eeg_vision_agreement", "command_latency_s",
                "latency_eeg_s", "latency_hybrid_s", "usable_trials",
            ):
                if key in row and row[key].strip():
                    val = float(row[key]) if key != "usable_trials" else int(float(row[key]))
                    d[key] = val
            if row.get("subjective_eeg_only", "").strip() or row.get("subjective_eeg_vision", "").strip():
                SUBJECTIVE_RATINGS[subj] = {
                    "eeg_only": float(row.get("subjective_eeg_only") or 0),
                    "eeg_vision": float(row.get("subjective_eeg_vision") or 0),
                }
    print(f"Loaded online metrics from {path.name}")


def load_offline_eeg() -> dict[str, float]:
    """Load latest offline EEGNet accuracy from table_i_data.json."""
    mapping = {"S1": "01", "S2": "02", "S3": "03", "S4": "04", "S10": "10"}
    offline: dict[str, float] = {}
    if not TABLE_JSON.exists():
        return {s: 0.0 for s in SUBJECTS}
    raw = json.loads(TABLE_JSON.read_text(encoding="utf-8"))
    for label, key in mapping.items():
        entry = raw.get(key, {}).get("EEGNet")
        offline[label] = entry["accuracy_pct"] if entry else 0.0
    return offline


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved: {path.relative_to(PAPER_DIR.parent)}")
    return path


# -----------------------------------------------------------------------------
# Fig 1 — Offline EEG vs Online EEG-only vs Online EEG+Vision (grouped bars)
# -----------------------------------------------------------------------------
def plot_offline_vs_online(offline: dict[str, float]) -> None:
    x = np.arange(len(SUBJECTS))
    w = 0.25

    off = [offline[s] for s in SUBJECTS]
    eeg = [SUBJECT_ONLINE_DATA[s]["online_eeg_success"] for s in SUBJECTS]
    hyb = [SUBJECT_ONLINE_DATA[s]["online_hybrid_success"] for s in SUBJECTS]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(x - w, off, w, label="Offline EEG (test acc.)", color=COLOR_OFFLINE, edgecolor="white")
    ax.bar(x, eeg, w, label="Online EEG-only (success)", color=COLOR_EEG, edgecolor="white")
    ax.bar(x + w, hyb, w, label="Online EEG+Vision (success)", color=COLOR_HYBRID, edgecolor="white")

    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(SUBJECTS)
    ax.set_ylabel("Accuracy / Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-Subject Performance: Offline Decoding vs Online Control")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "fig1_offline_vs_online.png")


# -----------------------------------------------------------------------------
# Fig 2 — EEG-only vs EEG+Vision side-by-side (online metrics only)
# -----------------------------------------------------------------------------
def plot_online_eeg_vs_vision() -> None:
    metrics = [
        ("online_eeg_success", "Control success (%)", "%"),
        ("online_hybrid_success", "Control success (%)", "%"),
        ("eeg_vision_agreement", "EEG–Vision agreement (%)", "%"),
    ]
    # We'll do 2 main bars per subject: eeg success vs hybrid success
    x = np.arange(len(SUBJECTS))
    w = 0.35

    eeg = [SUBJECT_ONLINE_DATA[s]["online_eeg_success"] for s in SUBJECTS]
    hyb = [SUBJECT_ONLINE_DATA[s]["online_hybrid_success"] for s in SUBJECTS]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    b1 = ax.bar(x - w / 2, eeg, w, label="EEG-only", color=COLOR_EEG, edgecolor="white")
    b2 = ax.bar(x + w / 2, hyb, w, label="EEG + Vision", color=COLOR_HYBRID, edgecolor="white")

    # Delta labels on top
    for i, (v1, v2) in enumerate(zip(eeg, hyb)):
        if v1 > 0 or v2 > 0:
            delta = v2 - v1
            sign = "+" if delta >= 0 else ""
            ax.annotate(
                f"{sign}{delta:.0f}",
                xy=(x[i], max(v1, v2) + 2),
                ha="center", fontsize=8,
                color=COLOR_DELTA_POS if delta >= 0 else COLOR_DELTA_NEG,
            )

    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBJECTS)
    ax.set_ylabel("Online Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Online Control: EEG-only vs EEG+Vision Hybrid")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "fig2_online_eeg_vs_vision.png")


# -----------------------------------------------------------------------------
# Fig 3 — Improvement delta (EEG+Vision − EEG-only) per subject
# -----------------------------------------------------------------------------
def plot_hybrid_improvement() -> None:
    deltas = [
        SUBJECT_ONLINE_DATA[s]["online_hybrid_success"] - SUBJECT_ONLINE_DATA[s]["online_eeg_success"]
        for s in SUBJECTS
    ]
    colors = [COLOR_DELTA_POS if d >= 0 else COLOR_DELTA_NEG for d in deltas]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(SUBJECTS, deltas, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ Success Rate (pp)\n(EEG+Vision − EEG-only)")
    ax.set_title("Hybrid Mode Improvement per Subject")
    ax.grid(axis="y", alpha=0.25)

    for bar, d in zip(bars, deltas):
        if d != 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                d + (1.5 if d >= 0 else -2.5),
                f"{d:+.1f}",
                ha="center", va="bottom" if d >= 0 else "top", fontsize=9,
            )
    _save(fig, "fig3_hybrid_improvement_delta.png")


# -----------------------------------------------------------------------------
# Fig 4 — Heatmap: subjects × metrics (normalized 0–100 for color)
# -----------------------------------------------------------------------------
def plot_performance_heatmap(offline: dict[str, float]) -> None:
    row_labels = SUBJECTS
    col_labels = [
        "Offline EEG\n(test acc.)",
        "Online EEG-only\n(success)",
        "Online EEG+Vision\n(success)",
        "EEG–Vision\nagreement",
    ]
    data = np.array([
        [
            offline[s],
            SUBJECT_ONLINE_DATA[s]["online_eeg_success"],
            SUBJECT_ONLINE_DATA[s]["online_hybrid_success"],
            SUBJECT_ONLINE_DATA[s]["eeg_vision_agreement"],
        ]
        for s in SUBJECTS
    ])

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=40, vmax=100)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_title("Performance Heatmap (%) — Green = Higher")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text = f"{val:.0f}" if val > 0 else "—"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Score (%)")
    _save(fig, "fig4_performance_heatmap.png")


# -----------------------------------------------------------------------------
# Fig 5 — EEG–Vision agreement vs hybrid success (scatter + trend)
# -----------------------------------------------------------------------------
def plot_agreement_vs_success() -> None:
    agree = [SUBJECT_ONLINE_DATA[s]["eeg_vision_agreement"] for s in SUBJECTS]
    hyb = [SUBJECT_ONLINE_DATA[s]["online_hybrid_success"] for s in SUBJECTS]
    eeg = [SUBJECT_ONLINE_DATA[s]["online_eeg_success"] for s in SUBJECTS]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(agree, hyb, s=120, c=COLOR_HYBRID, label="EEG+Vision success", zorder=3, edgecolors="white")
    ax.scatter(agree, eeg, s=80, c=COLOR_EEG, marker="s", label="EEG-only success", zorder=2, edgecolors="white")

    for i, s in enumerate(SUBJECTS):
        if agree[i] > 0:
            ax.annotate(s, (agree[i], hyb[i]), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="y = x")
    ax.set_xlabel("EEG–Vision Agreement (%)")
    ax.set_ylabel("Online Success Rate (%)")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_title("Does Vision Agreement Predict Hybrid Success?")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    _save(fig, "fig5_agreement_vs_success.png")


# -----------------------------------------------------------------------------
# Fig 6 — Command latency comparison (lower is better)
# -----------------------------------------------------------------------------
def plot_latency() -> None:
    # If you only have one latency per mode, extend dict; here single value per subject
    # interpreted as hybrid session mean — add eeg_only_latency to SUBJECT_ONLINE_DATA if needed
    lat_eeg = [SUBJECT_ONLINE_DATA[s].get("latency_eeg_s", SUBJECT_ONLINE_DATA[s]["command_latency_s"]) for s in SUBJECTS]
    lat_hyb = [SUBJECT_ONLINE_DATA[s].get("latency_hybrid_s", SUBJECT_ONLINE_DATA[s]["command_latency_s"]) for s in SUBJECTS]

    x = np.arange(len(SUBJECTS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x - w / 2, lat_eeg, w, label="EEG-only", color=COLOR_EEG)
    ax.bar(x + w / 2, lat_hyb, w, label="EEG+Vision", color=COLOR_HYBRID)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBJECTS)
    ax.set_ylabel("Mean Command Latency (s)")
    ax.set_title("Response Latency: EEG-only vs EEG+Vision")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "fig6_command_latency.png")


# -----------------------------------------------------------------------------
# Fig 7 — Subjective ratings (if provided)
# -----------------------------------------------------------------------------
def plot_subjective() -> None:
    if not SUBJECTIVE_RATINGS:
        return
    subjects = list(SUBJECTIVE_RATINGS.keys())
    eeg_r = [SUBJECTIVE_RATINGS[s].get("eeg_only", 0) for s in subjects]
    hyb_r = [SUBJECTIVE_RATINGS[s].get("eeg_vision", 0) for s in subjects]

    x = np.arange(len(subjects))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.bar(x - w / 2, eeg_r, w, label="EEG-only", color=COLOR_EEG)
    ax.bar(x + w / 2, hyb_r, w, label="EEG+Vision", color=COLOR_HYBRID)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylabel("User Rating (1–5)")
    ax.set_ylim(0, 5.5)
    ax.set_title("Subjective Usability Ratings")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "fig7_subjective_ratings.png")


# -----------------------------------------------------------------------------
# Fig 8 — Summary table figure (for paper / form appendix)
# -----------------------------------------------------------------------------
def plot_summary_table(offline: dict[str, float]) -> None:
    cols = ["Subject", "Offline\nEEG (%)", "Online\nEEG-only (%)", "Online\nEEG+Vis (%)", "Δ Hybrid\n(pp)", "EEG–Vis\nAgree (%)", "Trials"]
    rows = []
    for s in SUBJECTS:
        d = SUBJECT_ONLINE_DATA[s]
        delta = d["online_hybrid_success"] - d["online_eeg_success"]
        rows.append([
            s,
            f"{offline[s]:.1f}",
            f"{d['online_eeg_success']:.1f}" if d["online_eeg_success"] else "—",
            f"{d['online_hybrid_success']:.1f}" if d["online_hybrid_success"] else "—",
            f"{delta:+.1f}" if (d["online_eeg_success"] or d["online_hybrid_success"]) else "—",
            f"{d['eeg_vision_agreement']:.1f}" if d["eeg_vision_agreement"] else "—",
            str(d["usable_trials"]) if d["usable_trials"] else "—",
        ])

    fig, ax = plt.subplots(figsize=(8.5, 2.2 + 0.35 * len(SUBJECTS)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F2F2F2")
    ax.set_title("Per-Subject Evaluation Summary (fill online columns from your form)", pad=20, fontsize=11)
    _save(fig, "fig8_summary_table.png")


def main() -> None:
    load_from_csv()
    print("Loading offline EEG from table_i_data.json ...")
    offline = load_offline_eeg()
    print("Offline EEG (%):", offline)
    print("\nGenerating figures ...")
    plot_offline_vs_online(offline)
    plot_online_eeg_vs_vision()
    plot_hybrid_improvement()
    plot_performance_heatmap(offline)
    plot_agreement_vs_success()
    plot_latency()
    plot_subjective()
    plot_summary_table(offline)
    print(f"\nDone. All figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
