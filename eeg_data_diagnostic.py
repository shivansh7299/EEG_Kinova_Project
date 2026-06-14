"""
EEG Data Diagnostic (READ-ONLY)
================================
Helps explain why some subjects (e.g. 3 & 4) train to chance while others
(e.g. 1 & 2) reach 80%+.  It does NOT modify any data or the training pipeline.

It answers two questions raised in the code review:

  1) CHANNEL LAYOUT BUG CHECK
     The preprocessing notebooks select EEG channels with a hard-coded slice
     `data.columns[5:13]`.  That is only correct if there are exactly 5 columns
     BEFORE the first EEG electrode (Ch1 = Fz).  `dataCollection.py` currently
     writes only 4 metadata columns (trial_index, class_label, phase, timestamp)
     + Ch1..Ch17, so `5:13` would grab Ch2..Ch9 (drops Fz, adds an accel channel).
     This script prints, per subject, the column count and exactly which columns
     the slice `5:13` lands on -- so you can see if the layout differs between
     "good" and "bad" subjects.

  2) CLASS SEPARABILITY CHECK
     For the stimulus (MI) phase it computes, per class, the average 8-30 Hz band
     power per channel and plots it.  If the two classes look identical, the data
     itself carries little/no discriminative signal (a paradigm problem), which no
     model can fix.

Usage (run inside the project root with your training python/conda env):

    python eeg_data_diagnostic.py                  # compares subjects 1,2,3,4 (2_class)
    python eeg_data_diagnostic.py --subjects 2 4   # compare only subjects 2 and 4
    python eeg_data_diagnostic.py --classes 2

Outputs:
    - console report
    - PNG plots saved under  diagnostics/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # never pop a window; just save files
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

PROJECT_ROOT = Path(__file__).resolve().parent

# The 8 EEG electrodes the montage in the notebook expects (Ch1..Ch8).
EXPECTED_EEG_NAMES = ["Fz", "C3", "Cz", "C4", "P3", "Pz", "P4", "Oz"]

# What the preprocessing notebooks actually use:
NOTEBOOK_SLICE = slice(5, 13)   # data.columns[5:13]
# What real_time_eeg_predictor.py uses at inference (first 8 channels):
INFERENCE_SLICE_AFTER_META = slice(4, 12)  # 4 metadata cols then Ch1..Ch8

FS = 250
MI_BAND = (8.0, 30.0)  # motor-imagery band


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def find_subject_csvs(subject_no: int, n_class: int):
    """Return list of EEG_*.csv files for a subject, checking both folder styles."""
    candidates = [
        PROJECT_ROOT / "data" / f"Subject {subject_no}" / f"{n_class}_class",
        PROJECT_ROOT / "data" / f"Subject_{subject_no:02d}" / f"{n_class}_class",
        PROJECT_ROOT / "data" / f"Subject{subject_no}" / f"{n_class}_class",
    ]
    for folder in candidates:
        if folder.exists():
            files = sorted(folder.glob("EEG_*.csv"))
            if files:
                return folder, files
    return None, []


def load_subject(subject_no: int, n_class: int):
    folder, files = find_subject_csvs(subject_no, n_class)
    if not files:
        return None
    frames = []
    max_trial = 0
    for f in files:
        df = pd.read_csv(f)
        if "trial_index" in df.columns:
            df["trial_index"] = df["trial_index"] + max_trial
            max_trial = df["trial_index"].max() + 1
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    return {"folder": folder, "files": files, "data": data}


# --------------------------------------------------------------------------- #
# Check 1: column layout
# --------------------------------------------------------------------------- #
def report_columns(subject_no, info):
    data = info["data"]
    cols = list(data.columns)
    print("=" * 78)
    print(f"SUBJECT {subject_no}  ({len(info['files'])} file(s) in {info['folder']})")
    print("=" * 78)
    print(f"  total columns : {len(cols)}")
    print(f"  column names  : {cols}")

    def names_for(sl):
        return list(data.columns[sl])

    nb_cols = names_for(NOTEBOOK_SLICE)
    inf_cols = names_for(INFERENCE_SLICE_AFTER_META)

    print(f"\n  Preprocessing notebook uses columns[5:13]  -> {nb_cols}")
    print(f"  Real-time predictor uses first 8 (cols[4:12]) -> {inf_cols}")

    # Verdict
    looks_like_first8_after_4meta = nb_cols == [f"Ch{i}" for i in range(2, 10)]
    if nb_cols == [f"Ch{i}" for i in range(1, 9)]:
        print("  >> OK: notebook slice maps to Ch1..Ch8 (the real EEG electrodes).")
    elif looks_like_first8_after_4meta:
        print("  >> WARNING: notebook slice maps to Ch2..Ch9 -- it DROPS Ch1 (Fz) and "
              "INCLUDES Ch9 (a non-EEG/accelerometer channel).")
        print("     This is the channel-layout bug. It also disagrees with the "
              "real-time predictor (which uses Ch1..Ch8).")
    else:
        print("  >> NOTE: non-standard layout; inspect the column names above manually.")

    if nb_cols != inf_cols:
        print("  >> MISMATCH: training channels != inference channels "
              "(model will see different inputs live vs. during training).")
    print()
    return len(cols), nb_cols


# --------------------------------------------------------------------------- #
# Check 2: class separability
# --------------------------------------------------------------------------- #
def bandpass(sig, lo, hi, fs=FS, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def detect_eeg_columns(data):
    """Pick the EEG channel columns to ANALYZE for separability.

    We use the first 8 channel columns (Ch1..Ch8) because that is what the
    montage / hardware says are the real electrodes -- independent of the
    buggy 5:13 slice.
    """
    ch_cols = [c for c in data.columns if str(c).lower().startswith("ch")]
    if len(ch_cols) >= 8:
        return ch_cols[:8]
    # fall back to "after metadata" assumption
    return list(data.columns[4:12])


def class_bandpower(info, eeg_cols, phase="stimulus"):
    """Mean 8-30 Hz band power per class per channel during the given phase."""
    data = info["data"]
    if "phase" in data.columns:
        sub = data[data["phase"] == phase]
        if len(sub) == 0:  # phase label may differ; fall back to all rows
            sub = data
    else:
        sub = data
    sub = sub[sub["class_label"] != 0]

    results = {}
    for cls, grp in sub.groupby("class_label"):
        powers = []
        for ch in eeg_cols:
            x = grp[ch].to_numpy(dtype=float)
            if len(x) < FS:  # need at least ~1s
                powers.append(np.nan)
                continue
            try:
                xf = bandpass(x, *MI_BAND)
            except Exception:
                xf = x
            f, pxx = welch(xf, fs=FS, nperseg=min(512, len(xf)))
            mask = (f >= MI_BAND[0]) & (f <= MI_BAND[1])
            powers.append(float(np.mean(pxx[mask])) if mask.any() else np.nan)
        results[int(cls)] = np.array(powers)
    return results


def plot_separability(subject_no, eeg_cols, powers, out_dir):
    classes = sorted(powers.keys())
    if len(classes) < 2:
        return None
    x = np.arange(len(eeg_cols))
    width = 0.8 / len(classes)
    plt.figure(figsize=(10, 5))
    for i, cls in enumerate(classes):
        plt.bar(x + i * width, powers[cls], width, label=f"class {cls}")
    plt.xticks(x + width * (len(classes) - 1) / 2, eeg_cols, rotation=45)
    plt.ylabel(f"mean {MI_BAND[0]:.0f}-{MI_BAND[1]:.0f} Hz power")
    plt.title(f"Subject {subject_no}: per-class band power (stimulus phase)\n"
              "If the bars look the same across classes, the signal is not separable.")
    plt.legend()
    plt.tight_layout()
    out = out_dir / f"subject_{subject_no:02d}_class_bandpower.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def separability_index(powers):
    """Crude scalar: relative L1 difference between the two class power vectors."""
    classes = sorted(powers.keys())
    if len(classes) < 2:
        return None
    a, b = powers[classes[0]], powers[classes[1]]
    denom = (np.nanmean(np.abs(a)) + np.nanmean(np.abs(b))) / 2 + 1e-12
    return float(np.nanmean(np.abs(a - b)) / denom)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Read-only EEG data diagnostic.")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--classes", type=int, default=2)
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / "diagnostics"
    out_dir.mkdir(exist_ok=True)

    print("\n############ EEG DATA DIAGNOSTIC (read-only) ############\n")

    col_summary = {}
    sep_summary = {}

    for subj in args.subjects:
        info = load_subject(subj, args.classes)
        if info is None:
            print(f"SUBJECT {subj}: no EEG_*.csv found for {args.classes}_class -- skipped.\n")
            continue

        n_cols, nb_cols = report_columns(subj, info)
        col_summary[subj] = n_cols

        eeg_cols = detect_eeg_columns(info["data"])
        powers = class_bandpower(info, eeg_cols)
        sep = separability_index(powers)
        sep_summary[subj] = sep
        png = plot_separability(subj, eeg_cols, powers, out_dir)

        print(f"  EEG columns analyzed (Ch1..Ch8): {eeg_cols}")
        if sep is not None:
            print(f"  class-separability index (higher = more separable): {sep:.3f}")
        if png:
            print(f"  band-power plot saved: {png}")
        print()

    # ------------------------------------------------------------------ #
    print("=" * 78)
    print("CROSS-SUBJECT SUMMARY")
    print("=" * 78)
    if col_summary:
        counts = set(col_summary.values())
        print(f"  column counts per subject: {col_summary}")
        if len(counts) > 1:
            print("  >> RED FLAG: subjects have DIFFERENT column counts. The hard-coded")
            print("     columns[5:13] slice therefore selects DIFFERENT channels for")
            print("     different subjects -- a very likely cause of the accuracy gap.")
        else:
            print("  >> Column counts are consistent across subjects (channel-layout")
            print("     bug would then affect all subjects equally, not just 3 & 4).")
    if sep_summary:
        print(f"\n  class-separability index per subject: "
              f"{ {k: (round(v,3) if v is not None else None) for k,v in sep_summary.items()} }")
        print("  >> Compare good subjects (1,2) vs bad (3,4): a much lower index for")
        print("     3 & 4 means their two classes are nearly indistinguishable in the")
        print("     EEG -- i.e. a data/paradigm problem, not a model problem.")
    print(f"\nPlots written to: {out_dir}")
    print("\n############ DONE ############\n")


if __name__ == "__main__":
    main()
