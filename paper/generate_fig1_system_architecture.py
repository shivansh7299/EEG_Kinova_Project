"""
Generate system architecture PNG for the EEG-Kinova project.
Run: python paper/generate_fig1_system_architecture.py
Output: paper/figures/fig1_system_architecture.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "figures" / "fig1_system_architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "savefig.dpi": 300,
})

fig, ax = plt.subplots(figsize=(10.5, 7.0))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")

COLORS = {
    "sensor": "#D6E4F0",
    "proc": "#D5E8D4",
    "model": "#FFE6CC",
    "act": "#F8CECC",
    "gui": "#E1D5E7",
    "fusion": "#FFF2CC",
    "border": "#333333",
    "arrow": "#444444",
    "section_bg": "#F7F7F7",
}


def section_bg(x, y, w, h, title):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        facecolor=COLORS["section_bg"], edgecolor="#BBBBBB",
        linewidth=1.0, linestyle="-", zorder=0,
    )
    ax.add_patch(patch)
    ax.text(x + 0.15, y + h - 0.22, title, fontsize=9.5, fontweight="bold", color="#222222", va="top")


def box(x, y, w, h, text, color, fontsize=8, style="round", lw=1.2, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size=0.08" if style == "round" else "square,pad=0.02",
        facecolor=color, edgecolor=COLORS["border"], linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True, zorder=zorder + 1)


def arrow(x1, y1, x2, y2, style="-|>", lw=1.2, color=None, linestyle="-"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style,
            color=color or COLORS["arrow"],
            lw=lw,
            linestyle=linestyle,
            shrinkA=2,
            shrinkB=2,
        ),
        zorder=3,
    )


def line(x1, y1, x2, y2, lw=1.2, linestyle="-", color=None):
    ax.plot([x1, x2], [y1, y2], color=color or COLORS["arrow"], linewidth=lw, linestyle=linestyle, zorder=3)


# Title
ax.text(7.0, 9.65, "EEG-Kinova BCI System Architecture", ha="center", fontsize=13, fontweight="bold", color="#111111")

# ---------------------------------------------------------------------------
# OFFLINE PIPELINE
# ---------------------------------------------------------------------------
section_bg(0.3, 6.55, 13.4, 2.55, "Offline Pipeline (Training)")

box(0.55, 7.55, 1.75, 1.05, "Unicorn Hybrid Black\n8-ch EEG @ 250 Hz", COLORS["sensor"])
box(2.65, 7.55, 1.85, 1.05, "Data Collection\ndataCollection.py\nBaseline / Cue / MI", COLORS["proc"], fontsize=7.5)
box(4.85, 7.55, 1.85, 1.05, "Preprocessing\nNotebooks (2/3-class)\n4–40 Hz, z-score", COLORS["proc"], fontsize=7.5)
box(7.05, 7.55, 2.05, 1.05, "Model Training\nEEGNet / CTNet / FBMSNet", COLORS["model"], fontsize=7.5)
box(9.45, 7.55, 1.65, 1.05, "Checkpoints\n.pth + label_map", COLORS["model"], fontsize=7.5)
box(11.45, 7.55, 1.85, 1.05, "Results Store\nresults/Subject_XX/", COLORS["model"], fontsize=7.5)

arrow(2.3, 8.08, 2.65, 8.08)
arrow(4.5, 8.08, 4.85, 8.08)
arrow(6.7, 8.08, 7.05, 8.08)
arrow(9.1, 8.08, 9.45, 8.08)
arrow(11.1, 8.08, 11.45, 8.08)

gui = FancyBboxPatch(
    (3.2, 6.75), 7.6, 0.55,
    boxstyle="round,pad=0.02,rounding_size=0.08",
    facecolor=COLORS["gui"], edgecolor=COLORS["border"],
    linewidth=1.0, linestyle="--", zorder=1,
)
ax.add_patch(gui)
ax.text(7.0, 7.02, "Unified GUI: eeg_kinova_control_gui.py  →  Collect → Preprocess → Train → Deploy",
        ha="center", va="center", fontsize=7.8, style="italic")

for x in [4.2, 6.4, 8.6]:
    line(x, 7.3, x, 7.55, linestyle="--", lw=0.9)

# ---------------------------------------------------------------------------
# REAL-TIME SENSING
# ---------------------------------------------------------------------------
section_bg(0.3, 3.55, 13.4, 2.65, "Real-Time Control")

box(0.55, 4.55, 1.75, 1.0, "Streaming EEG\n250 Hz, 8 channels", COLORS["sensor"], fontsize=7.5)
box(0.55, 3.75, 1.75, 0.65, "USB Camera\n1280×720 @ 60 Hz", COLORS["sensor"], fontsize=7.5)

box(2.65, 4.35, 2.0, 1.2, "EEG Stream Buffer\nreal_time_eeg_predictor.py\n1 s window → 2 Hz inference", COLORS["proc"], fontsize=7.3)
box(2.65, 3.65, 2.0, 0.75, "Vision Pipeline\nHSV detect + Kalman\nBall / bar tracking", COLORS["proc"], fontsize=7.3)

box(5.1, 4.55, 2.15, 1.0, "EEG Prediction\nClass 0 = Up/Right\nClass 1 = Down/Left", COLORS["model"], fontsize=7.3)
box(5.1, 3.65, 2.15, 0.75, "Vision Output\nDirection (0/1) + speed\n|vx| from tracker", COLORS["model"], fontsize=7.3)

# Fusion block
box(7.75, 3.95, 2.55, 1.45, "Hybrid Fusion Logic\n(Option 6)\nEEG → direction (sign)\nVision → speed magnitude\nMatch → 100% speed\nMismatch → 50% speed", COLORS["fusion"], fontsize=7.2)

box(10.75, 4.05, 2.35, 1.25, "Kinova Gen3 Arm\nCartesian X velocity\nUp/Down bar control", COLORS["act"], fontsize=7.5)

# EEG path
arrow(2.3, 4.95, 2.65, 4.95)
arrow(4.65, 4.95, 5.1, 4.95)
arrow(7.25, 5.05, 7.75, 5.05)

# Vision path
arrow(2.3, 4.02, 2.65, 4.02)
arrow(4.65, 4.02, 5.1, 4.02)
arrow(7.25, 4.35, 7.75, 4.35)

# Fusion to arm
arrow(10.3, 4.68, 10.75, 4.68)

# Checkpoint to inference
line(10.28, 7.55, 10.28, 6.2, lw=1.3)
line(10.28, 6.2, 3.65, 6.2, lw=1.3)
arrow(3.65, 6.2, 3.65, 5.55, lw=1.3)
ax.text(10.45, 6.85, "load .pth", fontsize=7, color="#555555", rotation=90, va="center")

# EEG hardware to stream
line(1.42, 7.55, 1.42, 6.2, lw=1.3)
line(1.42, 6.2, 1.42, 5.55, lw=1.3)
arrow(1.42, 5.55, 1.42, 5.55)

# ---------------------------------------------------------------------------
# CONTROL MODES
# ---------------------------------------------------------------------------
section_bg(0.3, 0.35, 13.4, 2.85, "Deployment Modes")

box(0.55, 1.35, 3.0, 1.45, "Option 5: EEG-Only\nkinova_eeg_controller.py\n• No camera\n• EEG sets direction + speed\n• Position or velocity control", COLORS["proc"], fontsize=7.5)
box(4.0, 1.35, 3.35, 1.45, "Option 6: EEG + OpenCV\nkinova_eeg_opencv_controller.py\n• Camera calibration (4 corners)\n• 60 Hz vision + 2 Hz EEG\n• Velocity-driven paradigm", COLORS["fusion"], fontsize=7.5)
box(7.8, 1.35, 2.6, 1.45, "OpenCV Standalone\ntrack_ball_kinova_test_fixed.py\n• Vision-only bar tracking\n• No EEG required", COLORS["proc"], fontsize=7.3)
box(10.75, 1.35, 2.55, 1.45, "Shared Config\nhardware_config.py\nRobot IP, angles, V_MAX,\ncamera HSV, EEG sampling", COLORS["gui"], fontsize=7.3)

arrow(2.05, 2.8, 2.05, 3.55, lw=1.0)
arrow(5.65, 2.8, 8.0, 3.55, lw=1.0)
ax.text(2.05, 3.2, "deploy", fontsize=7, color="#555555", ha="center")
ax.text(6.8, 3.2, "deploy", fontsize=7, color="#555555", ha="center")

# Legend
legend_items = [
    (COLORS["sensor"], "Hardware / Sensors"),
    (COLORS["proc"], "Processing"),
    (COLORS["model"], "ML / Inference"),
    (COLORS["fusion"], "Fusion / Hybrid"),
    (COLORS["act"], "Actuator"),
]
lx, ly = 0.55, 0.55
for color, label in legend_items:
    patch = FancyBboxPatch(
        (lx, ly), 0.28, 0.22,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor=color, edgecolor=COLORS["border"], linewidth=0.8,
    )
    ax.add_patch(patch)
    ax.text(lx + 0.38, ly + 0.11, label, fontsize=7, va="center")
    lx += 2.35

fig.savefig(OUT, bbox_inches="tight", facecolor="white", pad_inches=0.15)
plt.close(fig)
print(f"Saved: {OUT}")
