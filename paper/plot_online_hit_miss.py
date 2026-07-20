"""
Plot EEG-only vs EEG+Vision online scores (Hit, Miss, Gain, Loss) per subject.

Your paper forms have:
  - EEG only   : trials 1-13
  - EEG+Vision : trials 1-14

HOW TO USE
----------
Option A — Edit SUBJECT_DATA below (quickest if you have totals per subject)

Option B — Fill paper/subject_online_scores.csv trial-by-trial (optional, for trial line plots)

Run:
  python paper/plot_online_hit_miss.py

Output:
  paper/figures/online_*.png
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAPER_DIR = Path(__file__).resolve().parent
FIG_DIR = PAPER_DIR / "figures"
CSV_PATH = PAPER_DIR / "subject_online_scores.csv"
TABLE_JSON = PAPER_DIR / "table_i_data.json"

OFFLINE_MODELS = ["EEGNet", "CTNet", "FBMSNet"]
OFFLINE_MODEL_COLORS = {
    "EEGNet": "#4C72B0",
    "CTNet": "#DD8452",
    "FBMSNet": "#55A868",
}

SUBJECTS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]  # Update with your subject IDs

# =============================================================================
# OPTION A — Paste totals from your paper forms here (per subject)
# =============================================================================
# Example (replace zeros with your handwritten values):
#   hit/miss/gain/loss = sum across all trials for that mode
#
SUBJECT_DATA = {
    "S1": {
        "EEG_only":   {"hit": 10, "miss": 6, "gain": 8, "loss": 6},
        "EEG_Vision": {"hit": 11, "miss": 6, "gain": 6, "loss": 3},
    },
    "S2": {
        "EEG_only":   {"hit":11, "miss": 5, "gain": 3, "loss": 5},
        "EEG_Vision": {"hit": 11, "miss": 5, "gain": 5, "loss": 5},
    },

    "S3": {
        "EEG_only":   {"hit": 8, "miss": 3, "gain": 3, "loss": 3},
        "EEG_Vision": {"hit": 10, "miss": 1, "gain": 3, "loss": 1},
    },
    "S4": {
        "EEG_only":   {"hit": 8, "miss": 4, "gain": 5, "loss": 4},
        "EEG_Vision": {"hit": 9, "miss": 3, "gain": 4, "loss": 3},
    },
    "S5": {
        "EEG_only":   {"hit": 13, "miss": 4, "gain": 4, "loss": 4},
        "EEG_Vision": {"hit": 10, "miss": 3, "gain": 4, "loss": 3},
    },
    "S6": {
        "EEG_only":   {"hit": 9, "miss": 5, "gain": 6, "loss": 5},
        "EEG_Vision": {"hit": 10, "miss": 3, "gain": 4, "loss": 3},
    },
    "S7": {
        "EEG_only":   {"hit": 9, "miss": 5, "gain": 5, "loss": 5},
        "EEG_Vision": {"hit": 9, "miss": 3, "gain": 4, "loss": 3},
    },
    "S8": {
        "EEG_only":   {"hit": 12, "miss": 3, "gain": 3, "loss": 3},
        "EEG_Vision": {"hit": 11, "miss": 2, "gain": 3, "loss": 2},
    },
    "S9": {
        "EEG_only":   {"hit": 12, "miss": 4, "gain": 5, "loss": 4},
        "EEG_Vision": {"hit": 11, "miss": 2, "gain": 4, "loss": 2},
    },
    "S10": {
        "EEG_only":   {"hit": 9, "miss": 4, "gain": 4, "loss": 4},
        "EEG_Vision": {"hit": 9, "miss": 2, "gain": 3, "loss": 2},
    },
}

# =============================================================================
# Plot style
# =============================================================================
COLOR_EEG = "#EE0855"
COLOR_VIS = "#55A868"
COLORS_METRIC = {"hit": "#4C72B0", "miss": "#C44E52", "gain": "#55A868", "loss": "#8172B2"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "savefig.dpi": 300,
})


def resolve_csv_path() -> Path | None:
    """Only use subject_online_scores.csv — never the template file."""
    return CSV_PATH if CSV_PATH.exists() else None


def load_trial_series(path: Path) -> dict | None:
    """Load cumulative Hit/Miss/Gain/Loss per trial for line plots."""
    if not path.exists():
        return None
    metrics = ("hit", "miss", "gain", "loss")
    series: dict = {
        s: {
            "EEG_only": {m: [] for m in metrics},
            "EEG_Vision": {m: [] for m in metrics},
        }
        for s in SUBJECTS
    }
    trials: dict = {s: {"EEG_only": [], "EEG_Vision": []} for s in SUBJECTS}
    found = False

    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            subj = row["subject"].strip()
            mode = row["mode"].strip()
            if subj not in series or mode not in series[subj]:
                continue
            try:
                trial = int(float(row.get("trial", 0) or 0))
            except ValueError:
                continue
            if trial <= 0:
                continue
            found = True
            trials[subj][mode].append(trial)
            for m in metrics:
                series[subj][mode][m].append(float(row.get(m, 0) or 0))

    if not found:
        return None

    # Sort each mode by trial number (CSV may be out of order)
    for subj in SUBJECTS:
        for mode in ("EEG_only", "EEG_Vision"):
            order = np.argsort(trials[subj][mode])
            trials[subj][mode] = [trials[subj][mode][i] for i in order]
            for m in metrics:
                vals = series[subj][mode][m]
                series[subj][mode][m] = [vals[i] for i in order]
    series["_trials"] = trials
    return series


def load_from_csv(path: Path) -> dict | None:
    """Totals per subject/mode — use last trial row (values are cumulative)."""
    series = load_trial_series(path)
    if not series:
        return None
    metrics = ("hit", "miss", "gain", "loss")
    data = {
        s: {
            "EEG_only": {m: 0 for m in metrics},
            "EEG_Vision": {m: 0 for m in metrics},
        }
        for s in SUBJECTS
    }
    for subj in SUBJECTS:
        for mode in ("EEG_only", "EEG_Vision"):
            for m in metrics:
                vals = series[subj][mode][m]
                data[subj][mode][m] = vals[-1] if vals else 0
    return data


def success_rate(d: dict) -> float:
    total = d["hit"] + d["miss"]
    return (d["hit"] / total * 100) if total > 0 else 0.0


def net_score(d: dict) -> float:
    """Hit+Gain positive, Miss+Loss negative."""
    return d["hit"] + d["gain"] - d["miss"] - d["loss"]

def net_score_diff(d: dict) -> float:
    """Hit+Gain positive, Miss+Loss negative."""
    return d["hit"] - d["miss"] + d["gain"] - d["loss"]


def get_data() -> dict:
    csv_path = resolve_csv_path()
    if csv_path:
        csv_data = load_from_csv(csv_path)
        if csv_data:
            print(f"Loaded trial data from {csv_path.name}")
            return csv_data
    print("Using SUBJECT_DATA dict in script")
    return SUBJECT_DATA


def get_trial_series() -> dict | None:
    csv_path = resolve_csv_path()
    if not csv_path:
        return None
    return load_trial_series(csv_path)


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PAPER_DIR.parent)}")


def plot_success_rate(data: dict) -> None:
    """Fig 1 — Hit rate = Hit / (Hit + Miss) for EEG vs EEG+Vision."""
    x = np.arange(len(SUBJECTS))
    w = 0.35
    eeg = [success_rate(data[s]["EEG_only"]) for s in SUBJECTS]
    vis = [success_rate(data[s]["EEG_Vision"]) for s in SUBJECTS]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, eeg, w, label="EEG only", color=COLOR_EEG, edgecolor="white")
    ax.bar(x + w / 2, vis, w, label="EEG + Vision", color=COLOR_VIS, edgecolor="white")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBJECTS)
    ax.set_ylabel("Success Rate (%)  = Hit / (Hit + Miss)")
    ax.set_ylim(0, 105)
    ax.set_title("Online Control Success Rate per Subject")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "online_fig1_success_rate.png")


def plot_hit_miss_stacked(data: dict) -> None:
    """Fig 2 — Stacked Hit vs Miss (two panels: EEG only | EEG+Vision)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    modes = [("EEG_only", "EEG Only", COLOR_EEG), ("EEG_Vision", "EEG + Vision", COLOR_VIS)]

    for ax, (mode_key, title, _) in zip(axes, modes):
        hits = [data[s][mode_key]["hit"] for s in SUBJECTS]
        misses = [data[s][mode_key]["miss"] for s in SUBJECTS]
        x = np.arange(len(SUBJECTS))
        ax.bar(x, hits, label="Hit", color=COLORS_METRIC["hit"])
        ax.bar(x, misses, bottom=hits, label="Miss", color=COLORS_METRIC["miss"])
        ax.set_xticks(x)
        ax.set_xticklabels(SUBJECTS)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Count (trials)")
    axes[1].legend(loc="upper right")
    fig.suptitle("Hit vs Miss Counts per Subject", y=1.02)
    fig.tight_layout()
    _save(fig, "online_fig2_hit_miss_stacked.png")


def plot_all_metrics_grouped(data: dict) -> None:
    """Fig 3 — Hit, Miss, Gain, Loss grouped bars: EEG vs EEG+Vision per subject."""
    metrics = ["hit", "miss", "gain", "loss"]
    n_subj = len(SUBJECTS)
    ncols = 3
    nrows = int(np.ceil(n_subj / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.0 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, subj in enumerate(SUBJECTS):
        ax = axes[i]
        eeg_vals = [data[subj]["EEG_only"][m] for m in metrics]
        vis_vals = [data[subj]["EEG_Vision"][m] for m in metrics]
        x = np.arange(len(metrics))
        w = 0.35
        ax.bar(x - w / 2, eeg_vals, w, label="EEG", color=COLOR_EEG)
        ax.bar(x + w / 2, vis_vals, w, label="EEG+Vis", color=COLOR_VIS)
        ax.set_xticks(x)
        ax.set_xticklabels(["Hit", "Miss", "Gain", "Loss"])
        ax.set_title(subj)
        ax.grid(axis="y", alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8)

    if len(SUBJECTS) < len(axes):
        for j in range(len(SUBJECTS), len(axes)):
            axes[j].axis("off")

    fig.suptitle("Hit / Miss / Gain / Loss — EEG vs EEG+Vision", y=1.01)
    fig.tight_layout()
    _save(fig, "online_fig3_all_metrics_by_subject.png")


# def plot_hybrid_improvement(data: dict) -> None:
#     """Fig 4 — Change in success rate: EEG+Vision minus EEG-only."""
#     deltas = [
#         success_rate(data[s]["EEG_Vision"]) - success_rate(data[s]["EEG_only"])
#         for s in SUBJECTS
#     ]
#     colors = ["#55A868" if d >= 0 else "#C44E52" for d in deltas]

#     fig, ax = plt.subplots(figsize=(6.5, 4))
#     bars = ax.bar(SUBJECTS, deltas, color=colors, edgecolor="white")
#     ax.axhline(0, color="black", linewidth=0.8)
#     ax.set_ylabel("Δ Success Rate (pp)\n(EEG+Vision − EEG only)")
#     ax.set_title("Hybrid Mode Improvement per Subject")
#     ax.grid(axis="y", alpha=0.25)
#     for bar, d in zip(bars, deltas):
#         if d != 0:
#             ax.text(bar.get_x() + bar.get_width() / 2, d + (0.8 if d >= 0 else -1.2),
#                     f"{d:+.1f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=9)
#     _save(fig, "online_fig4_improvement_delta.png")


def plot_hybrid_improvement(data: dict) -> None:
    """Fig 4 — Change in success rate: EEG+Vision minus EEG-only."""
    deltas = [
        success_rate(data[s]["EEG_Vision"]) - success_rate(data[s]["EEG_only"])
        for s in SUBJECTS
    ]
    colors = ["#55A868" if d >= 0 else "#C44E52" for d in deltas]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(SUBJECTS, deltas, color=colors, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ Success Rate (pp)\n(EEG+Vision − EEG only)")
    ax.set_title("Hybrid Mode Improvement per Subject", pad=12)
    ax.grid(axis="y", alpha=0.25)

    # Add extra space above tallest bar so labels do not go outside
    ymax = max(deltas)
    ymin = min(0, min(deltas))
    ax.set_ylim(ymin, ymax + 4)

    for bar, d in zip(bars, deltas):
        if d != 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                d + 0.6 if d >= 0 else d - 0.6,
                f"{d:+.1f}",
                ha="center",
                va="bottom" if d >= 0 else "top",
                fontsize=9
            )

    fig.tight_layout()
    _save(fig, "online_fig4_improvement_delta.png")


def plot_net_score(data: dict) -> None:
    """Fig 5 — Net score = (Hit+Gain) − (Miss+Loss)."""
    x = np.arange(len(SUBJECTS))
    w = 0.35
    eeg = [net_score(data[s]["EEG_only"]) for s in SUBJECTS]
    vis = [net_score(data[s]["EEG_Vision"]) for s in SUBJECTS]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, eeg, w, label="EEG only", color=COLOR_EEG)
    ax.bar(x + w / 2, vis, w, label="EEG + Vision", color=COLOR_VIS)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBJECTS)
    ax.set_ylabel("Net Score  (Hit+Gain) − (Miss+Loss)")
    ax.set_title("Overall Online Performance Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "online_fig5_net_score.png")

def plot_net_score_diff(data: dict) -> None:
    """Fig 6 — Net Score  (Hit+Miss) - (Gain+Loss)."""
    x = np.arange(len(SUBJECTS))
    w = 0.35
    eeg = [net_score_diff(data[s]["EEG_only"]) for s in SUBJECTS]
    vis = [net_score_diff(data[s]["EEG_Vision"]) for s in SUBJECTS]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, eeg, w, label="EEG only", color=COLOR_EEG)
    ax.bar(x + w / 2, vis, w, label="EEG + Vision", color=COLOR_VIS)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBJECTS)
    ax.set_ylabel("Net Score  (Hit+Miss) − (Gain+Loss)")
    ax.set_title("Overall Online Performance Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "online_fig5_net_score_diff.png")


def plot_metric_lines_by_subject(series: dict, metric: str, title: str) -> None:
    """Line graph: cumulative metric over trials, EEG vs EEG+Vision per subject."""
    n_subj = len(SUBJECTS)
    ncols = 3
    nrows = int(np.ceil(n_subj / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.2 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, subj in enumerate(SUBJECTS):
        ax = axes[i]
        for mode, label, color in (
            ("EEG_only", "EEG only", COLOR_EEG),
            ("EEG_Vision", "EEG + Vision", COLOR_VIS),
        ):
            trials = series["_trials"][subj][mode]
            vals = series[subj][mode][metric]
            if trials and vals:
                ax.plot(trials, vals, marker="o", markersize=4, linewidth=1.8, label=label, color=color)
        ax.set_title(subj, fontsize=10, fontweight="bold")
        ax.set_xlabel("Trial")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_ylabel(f"Cumulative {title}")
            ax.legend(fontsize=8)

    for j in range(n_subj, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{title} over Trials — EEG only vs EEG + Vision", y=1.01, fontsize=12)
    fig.tight_layout()
    _save(fig, f"online_line_{metric}.png")


def plot_all_metrics_lines_mean(series: dict) -> None:
    """2×2 line graph: mean cumulative Hit/Miss/Gain/Loss across subjects."""
    metrics = [
        ("hit", "Hit"),
        ("miss", "Miss"),
        ("gain", "Gain"),
        ("loss", "Loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=False)
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        for mode, label, color in (
            ("EEG_only", "EEG only", COLOR_EEG),
            ("EEG_Vision", "EEG + Vision", COLOR_VIS),
        ):
            max_trial = 0
            aligned = []
            for subj in SUBJECTS:
                trials = series["_trials"][subj][mode]
                vals = series[subj][mode][metric]
                if not trials:
                    continue
                max_trial = max(max_trial, max(trials))
                trial_to_val = dict(zip(trials, vals))
                aligned.append(trial_to_val)

            if not aligned or max_trial == 0:
                continue
            rows = []
            for trial_to_val in aligned:
                last = 0.0
                row = []
                for t in range(1, max_trial + 1):
                    if t in trial_to_val:
                        last = trial_to_val[t]
                    row.append(last)
                rows.append(row)
            arr = np.array(rows)
            x = np.arange(1, max_trial + 1)
            mean = arr.mean(axis=0)
            ax.plot(x, mean, marker="o", markersize=3, linewidth=2.0, label=label, color=color)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Trial")
        ax.set_ylabel(f"Cumulative {title}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Mean Cumulative Scores — EEG only vs EEG + Vision", y=1.01, fontsize=12)
    fig.tight_layout()
    _save(fig, "online_line_all_metrics.png")


def plot_metrics_across_subjects(data: dict) -> None:
    """Line graphs: subjects on x-axis, total score on y-axis, EEG vs EEG+Vision."""
    metrics = [
        ("hit", "Hit", COLORS_METRIC["hit"]),
        ("miss", "Miss", COLORS_METRIC["miss"]),
        ("gain", "Gain", COLORS_METRIC["gain"]),
        ("loss", "Loss", COLORS_METRIC["loss"]),
    ]
    x = np.arange(len(SUBJECTS))

    for metric, title, accent in metrics:
        eeg_vals = [data[s]["EEG_only"][metric] for s in SUBJECTS]
        vis_vals = [data[s]["EEG_Vision"][metric] for s in SUBJECTS]

        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(x, eeg_vals, marker="o", markersize=7, linewidth=2.2, label="EEG only", color=COLOR_EEG)
        ax.plot(x, vis_vals, marker="s", markersize=7, linewidth=2.2, label="EEG + Vision", color=COLOR_VIS)

        for i, (e, v) in enumerate(zip(eeg_vals, vis_vals)):
            if e != v:
                ax.annotate(
                    f"{v - e:+.0f}",
                    xy=(i, max(e, v)),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="#333333",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(SUBJECTS)
        ax.set_xlabel("Subject")
        ax.set_ylabel(f"{title} Score")
        ax.set_title(f"{title} — EEG only vs EEG + Vision (All Subjects)")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        ymax = max(max(eeg_vals), max(vis_vals), 1)
        ax.set_ylim(0, ymax * 1.15)
        _save(fig, f"online_line_by_subject_{metric}.png")

    # Combined 2×2 panel
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()
    for ax, (metric, title, _) in zip(axes, metrics):
        eeg_vals = [data[s]["EEG_only"][metric] for s in SUBJECTS]
        vis_vals = [data[s]["EEG_Vision"][metric] for s in SUBJECTS]
        ax.plot(x, eeg_vals, marker="o", markersize=6, linewidth=2.0, label="EEG only", color=COLOR_EEG)
        ax.plot(x, vis_vals, marker="s", markersize=6, linewidth=2.0, label="EEG + Vision", color=COLOR_VIS)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(f"{title} Score")
        ax.grid(alpha=0.25)
        ymax = max(max(eeg_vals), max(vis_vals), 1)
        ax.set_ylim(0, ymax * 1.15)
        if ax is axes[-1] or ax is axes[-2]:
            ax.set_xlabel("Subject")
        if ax is axes[0]:
            ax.legend(fontsize=8)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(SUBJECTS)
    axes[-2].set_xticks(x)
    axes[-2].set_xticklabels(SUBJECTS)
    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels(SUBJECTS)

    fig.suptitle("Hit / Miss / Gain / Loss by Subject — EEG only vs EEG + Vision", y=1.01, fontsize=12)
    fig.tight_layout()
    _save(fig, "online_line_by_subject_all_metrics.png")


def plot_metric_lines_combined(series: dict) -> None:
    """Four-panel line graph (per subject subplots) in one tall figure."""
    metrics = [("hit", "Hit"), ("miss", "Miss"), ("gain", "Gain"), ("loss", "Loss")]
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=False)

    for ax, (metric, title) in zip(axes, metrics):
        for subj in SUBJECTS:
            for mode, color, ls in (
                ("EEG_only", COLOR_EEG, "-"),
                ("EEG_Vision", COLOR_VIS, "--"),
            ):
                trials = series["_trials"][subj][mode]
                vals = series[subj][mode][metric]
                if trials and vals:
                    ax.plot(
                        trials, vals,
                        linewidth=1.4, linestyle=ls, color=color, alpha=0.55,
                    )
        # Legend proxies: thick lines for modes
        ax.plot([], [], color=COLOR_EEG, linewidth=2.5, label="EEG only")
        ax.plot([], [], color=COLOR_VIS, linewidth=2.5, linestyle="--", label="EEG + Vision")
        ax.set_ylabel(f"Cumulative {title}")
        ax.set_title(title, fontweight="bold", loc="left")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")

    axes[-1].set_xlabel("Trial")
    fig.suptitle("Hit / Miss / Gain / Loss — All Subjects (EEG vs EEG+Vision)", y=1.01, fontsize=12)
    fig.tight_layout()
    _save(fig, "online_line_combined_4panel.png")


def plot_summary_table(data: dict) -> None:
    """Fig 6 — Printable summary table for paper appendix."""
    cols = ["Subject", "Mode", "Hit", "Miss", "Gain", "Loss", "Success %", "Net"]
    rows = []
    for s in SUBJECTS:
        for mode, label in [("EEG_only", "EEG only"), ("EEG_Vision", "EEG+Vis")]:
            d = data[s][mode]
            rows.append([
                s, label,
                str(int(d["hit"])), str(int(d["miss"])),
                str(int(d["gain"])), str(int(d["loss"])),
                f"{success_rate(d):.1f}", f"{net_score(d):+.0f}",
            ])

    fig, ax = plt.subplots(figsize=(9, 2.5 + 0.3 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", weight="bold")
    ax.set_title("Online Evaluation Summary (from paper forms)", pad=16, fontsize=11)
    _save(fig, "online_fig6_summary_table.png")


def online_label_to_table_key(label: str) -> str:
    return f"{int(label.replace('S', '')):02d}"


def load_offline_table() -> dict:
    """Latest offline test accuracy per subject/model from Table I builder."""
    sys.path.insert(0, str(PAPER_DIR))
    try:
        from build_table_i import build_table

        return build_table()
    except Exception:
        if TABLE_JSON.exists():
            return json.loads(TABLE_JSON.read_text(encoding="utf-8"))
        return {}


def plot_offline_accuracy_by_subject() -> None:
    """Line graph: offline test accuracy (%) per subject, one line per model."""
    table = load_offline_table()
    if not table:
        print("  skip offline accuracy graph — no table_i_data.json and build_table_i failed")
        return

    x = np.arange(len(SUBJECTS))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    for model in OFFLINE_MODELS:
        ys = []
        for subj in SUBJECTS:
            key = online_label_to_table_key(subj)
            entry = table.get(key, {}).get(model)
            ys.append(entry["accuracy_pct"] if entry else np.nan)
        ax.plot(
            x, ys,
            marker="o", markersize=7, linewidth=2.2,
            label=model, color=OFFLINE_MODEL_COLORS[model],
        )
        for i, y in enumerate(ys):
            if not np.isnan(y):
                ax.annotate(
                    f"{y:.1f}",
                    xy=(i, y),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=OFFLINE_MODEL_COLORS[model],
                )

    ax.axhline(50, color="gray", linestyle="--", linewidth=0.9, alpha=0.7, label="Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(SUBJECTS)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Offline Test Accuracy (%)")
    ax.set_title("Latest Offline Accuracy by Subject and Model (Table I)")
    ax.set_ylim(40, 100)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "online_offline_accuracy_by_subject.png")


def print_summary(data: dict) -> None:
    print("\n" + "=" * 72)
    print("ONLINE SCORES SUMMARY")
    print("=" * 72)
    print(f"{'Subj':<6} {'Mode':<12} {'Hit':>5} {'Miss':>5} {'Gain':>5} {'Loss':>5} {'Succ%':>7} {'Net':>6}")
    print("-" * 72)
    for s in SUBJECTS:
        for mode, label in [("EEG_only", "EEG only"), ("EEG_Vision", "EEG+Vis")]:
            d = data[s][mode]
            print(f"{s:<6} {label:<12} {d['hit']:5.0f} {d['miss']:5.0f} {d['gain']:5.0f} {d['loss']:5.0f} "
                  f"{success_rate(d):7.1f} {net_score(d):6.0f}")
    print("=" * 72)


def main() -> None:
    data = get_data()
    if all(sum(data[s][m].values()) == 0 for s in SUBJECTS for m in ("EEG_only", "EEG_Vision")):
        print("WARNING: All values are still 0. Fill SUBJECT_DATA or CSV from your paper forms.\n")

    print_summary(data)
    print("\nGenerating figures...")
    plot_success_rate(data)
    plot_hit_miss_stacked(data)
    plot_all_metrics_grouped(data)
    plot_hybrid_improvement(data)
    plot_net_score(data)
    plot_net_score_diff(data)
    plot_summary_table(data)
    print("\nGenerating subject-comparison line graphs (x = subject)...")
    plot_metrics_across_subjects(data)

    series = get_trial_series()
    if series:
        print("\nGenerating line graphs (trial-by-trial)...")
        for metric, title in [("hit", "Hit"), ("miss", "Miss"), ("gain", "Gain"), ("loss", "Loss")]:
            plot_metric_lines_by_subject(series, metric, title)
        plot_all_metrics_lines_mean(series)
        plot_metric_lines_combined(series)
    else:
        print("\nSkipping trial-by-trial line graphs — subject_online_scores.csv not found")
        print("  (subject-comparison graphs use SUBJECT_DATA totals)")

    print("\nGenerating offline accuracy line graph (Table I)...")
    plot_offline_accuracy_by_subject()

    print(f"\nDone. Figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
