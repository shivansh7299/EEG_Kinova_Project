"""
Plot post-experiment questionnaire responses.

Input:
  Questinair response.xlsx

Outputs:
  paper/figures/questionnaire_*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = Path(__file__).resolve().parent
FIG_DIR = PAPER_DIR / "figures"
XLSX_PATH = PROJECT_ROOT / "Questinair response.xlsx"

QUESTION_TEXT = {
    "Q1": "Vision improved overall control experience",
    "Q2": "Arm accuracy with EEG only",
    "Q3": "Arm accuracy with EEG + Vision",
    "Q4": "EEG + Vision felt better than EEG only",
    "Q5": "Confidence with EEG only",
    "Q6": "Confidence with EEG + Vision",
    "Q7": "Mental effortlessness with EEG + Vision",
}

SHORT_LABELS = {
    "Q1": "Vision\nimproved\ncontrol",
    "Q2": "Accuracy\nEEG only",
    "Q3": "Accuracy\nEEG+Vision",
    "Q4": "Hybrid\nfelt better",
    "Q5": "Confidence\nEEG only",
    "Q6": "Confidence\nEEG+Vision",
    "Q7": "Effortless\nEEG+Vision",
}

COLOR_EEG = "#DD8452"
COLOR_VIS = "#55A868"
COLOR_BAR = "#4C72B0"
COLOR_GRID = "#D0D0D0"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "savefig.dpi": 300,
    }
)


def load_data() -> pd.DataFrame:
    if not XLSX_PATH.exists():
        raise FileNotFoundError(f"Questionnaire response file not found: {XLSX_PATH}")

    df = pd.read_excel(XLSX_PATH)
    df = df.rename(columns=lambda c: str(c).strip())
    question_cols = [c for c in QUESTION_TEXT if c in df.columns]
    if not question_cols:
        raise ValueError("No Q1-Q7 columns found in questionnaire workbook.")

    df[question_cols] = df[question_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=question_cols, how="all")
    return df


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out.relative_to(PROJECT_ROOT)}")


def plot_mean_scores(df: pd.DataFrame) -> None:
    questions = list(QUESTION_TEXT)
    means = df[questions].mean()
    stds = df[questions].std()
    x = np.arange(len(questions))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=COLOR_BAR, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[q] for q in questions])
    ax.set_ylabel("Score (0-6)")
    ax.set_ylim(0, 6.5)
    ax.set_title("Post-Experiment Questionnaire: Mean Scores")
    ax.grid(axis="y", color=COLOR_GRID, alpha=0.5)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.12,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _save(fig, "questionnaire_mean_scores.png")


def plot_subject_lines(df: pd.DataFrame) -> None:
    questions = list(QUESTION_TEXT)
    x = np.arange(len(questions))

    fig, ax = plt.subplots(figsize=(9, 5))
    for _, row in df.iterrows():
        subject = int(row["Subject No"]) if "Subject No" in df.columns else int(row.name + 1)
        ax.plot(
            x,
            row[questions].values,
            marker="o",
            linewidth=1.2,
            alpha=0.65,
            label=f"S{subject}",
        )

    ax.plot(x, df[questions].mean().values, color="black", marker="s", linewidth=2.5, label="Mean")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i}" for i in range(1, len(questions) + 1)])
    ax.set_ylabel("Score (0-6)")
    ax.set_ylim(0, 6.5)
    ax.set_title("Questionnaire Scores by Subject")
    ax.grid(axis="y", color=COLOR_GRID, alpha=0.5)
    ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.35))
    _save(fig, "questionnaire_subject_lines.png")


def plot_eeg_vs_vision_pairs(df: pd.DataFrame) -> None:
    pairs = [
        ("Q2", "Q3", "Arm accuracy"),
        ("Q5", "Q6", "Confidence"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4), sharey=True)
    for ax, (q_eeg, q_vis, title) in zip(axes, pairs):
        eeg = df[q_eeg].to_numpy(dtype=float)
        vis = df[q_vis].to_numpy(dtype=float)
        x = np.array([0, 1])

        for e, v in zip(eeg, vis):
            ax.plot(x, [e, v], color="#999999", marker="o", alpha=0.55, linewidth=1.2)

        means = [np.nanmean(eeg), np.nanmean(vis)]
        ax.bar(x, means, width=0.45, color=[COLOR_EEG, COLOR_VIS], alpha=0.85, edgecolor="white")
        for xi, mean in zip(x, means):
            ax.text(xi, mean + 0.12, f"{mean:.2f}", ha="center", va="bottom", fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels(["EEG only", "EEG+Vision"])
        ax.set_title(title)
        ax.set_ylim(0, 6.5)
        ax.grid(axis="y", color=COLOR_GRID, alpha=0.5)

    axes[0].set_ylabel("Score (0-6)")
    fig.suptitle("EEG Only vs EEG + Vision Questionnaire Ratings", y=1.02)
    fig.tight_layout()
    _save(fig, "questionnaire_eeg_vs_vision_pairs.png")


def plot_heatmap(df: pd.DataFrame) -> None:
    questions = list(QUESTION_TEXT)
    values = df[questions].to_numpy(dtype=float)
    subjects = (
        [f"S{int(v)}" for v in df["Subject No"].values]
        if "Subject No" in df.columns
        else [f"S{i + 1}" for i in range(len(df))]
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=6)
    ax.set_xticks(np.arange(len(questions)))
    ax.set_xticklabels([f"Q{i}" for i in range(1, len(questions) + 1)])
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_yticklabels(subjects)
    ax.set_xlabel("Question")
    ax.set_ylabel("Subject")
    ax.set_title("Questionnaire Response Heatmap")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center", color="black", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Score (0-6)")
    _save(fig, "questionnaire_response_heatmap.png")


def print_summary(df: pd.DataFrame) -> None:
    questions = list(QUESTION_TEXT)
    print("\nQUESTIONNAIRE SUMMARY")
    print("=" * 72)
    print(f"{'Question':<8} {'Mean':>7} {'Std':>7} {'Median':>8}  Text")
    print("-" * 72)
    for q in questions:
        print(
            f"{q:<8} {df[q].mean():7.2f} {df[q].std():7.2f} {df[q].median():8.2f}  "
            f"{QUESTION_TEXT[q]}"
        )
    print("=" * 72)


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df)} questionnaire responses from {XLSX_PATH.name}")
    print_summary(df)
    plot_mean_scores(df)
    plot_subject_lines(df)
    plot_eeg_vs_vision_pairs(df)
    plot_heatmap(df)


if __name__ == "__main__":
    main()
