"""
EEG Windowed Cross-Validated Separability (READ-ONLY)
=====================================================
Answers: "Is there a reliable, generalizable difference between the two classes
in this subject's EEG?" -- now using the SAME windowing/overlap scheme as the
EEGNet pipeline, evaluated with four classic ML models.

Pipeline
--------
1. Take the stimulus (motor-imagery) phase of each trial, channels Ch1..Ch8.
2. Slide overlapping windows over each trial (default: 250-sample window,
   75-sample step = 70% overlap, exactly like EEGNet training).
3. For every window compute 16 features = log band-power (mu 8-12, beta 13-30 Hz)
   per channel.  Each window becomes one labelled sample.
4. Cross-validate four models: LDA, Logistic Regression, SVM (RBF), Random Forest.

Two CV numbers are reported per model:
  * GROUPED (by trial)  -> windows from the same trial stay in the SAME fold.
       This is leak-free and TRUSTWORTHY -- it measures generalization to
       unseen trials.
  * LEAKY (window-shuffled) -> windows shuffled across folds, ignoring which
       trial they came from.  This mimics EEGNet's train/val split (which leaks
       windows of the same trial across the split) and is OPTIMISTIC.  Use it
       only to understand why a windowed deep model can look better than it is.

It does NOT modify any data or the training pipeline.

Usage:
    python eeg_separability_cv.py
    python eeg_separability_cv.py --subjects 1 2 3 4 --classes 2
    python eeg_separability_cv.py --window 250 --step 75
    python eeg_separability_cv.py --no-window     # one feature vector per trial
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, GroupKFold

try:  # available in newer scikit-learn; gives balanced + grouped folds
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except ImportError:
    HAS_SGKF = False

PROJECT_ROOT = Path(__file__).resolve().parent

FS = 250
EEG_CHANNELS = [f"Ch{i}" for i in range(1, 9)]   # Ch1..Ch8 = Fz,C3,Cz,C4,P3,Pz,P4,Oz
BANDS = {"mu": (8.0, 12.0), "beta": (13.0, 30.0)}
STIM_PHASE = "stimulus"


def find_subject_csvs(subject_no, n_class):
    for folder in [
        PROJECT_ROOT / "data" / f"Subject {subject_no}" / f"{n_class}_class",
        PROJECT_ROOT / "data" / f"Subject_{subject_no:02d}" / f"{n_class}_class",
    ]:
        if folder.exists():
            files = sorted(folder.glob("EEG_*.csv"))
            if files:
                return folder, files
    return None, []


def load_subject(subject_no, n_class):
    folder, files = find_subject_csvs(subject_no, n_class)
    if not files:
        return None
    frames, max_trial = [], 0
    for f in files:
        df = pd.read_csv(f)
        if "trial_index" in df.columns:
            df["trial_index"] = df["trial_index"] + max_trial
            max_trial = df["trial_index"].max() + 1
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def bandpass(sig, lo, hi, fs=FS, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def segment_features(seg):
    """seg: (L, n_channels). Return 16 features = log band-power per channel/band."""
    feats = []
    for ci in range(seg.shape[1]):
        x = seg[:, ci]
        for (lo, hi) in BANDS.values():
            try:
                xf = bandpass(x, lo, hi)
            except Exception:
                xf = x
            feats.append(np.log(np.var(xf) + 1e-8))
    return feats


def build_dataset(data, window=None, step=None):
    """Build (X, y, groups).

    If window is None  -> one feature vector per trial (whole stimulus phase).
    If window is given -> overlapping windows per trial (EEGNet-style); each
                          window is a sample and `groups` holds its trial id so
                          CV can keep same-trial windows together.
    """
    if "phase" in data.columns and (data["phase"] == STIM_PHASE).any():
        data = data[data["phase"] == STIM_PHASE]
    data = data[data["class_label"] != 0]

    X, y, groups = [], [], []
    for trial_id, grp in data.groupby("trial_index"):
        labels = grp["class_label"].unique()
        if len(labels) != 1:
            continue
        if any(ch not in grp.columns for ch in EEG_CHANNELS):
            continue
        mat = grp[EEG_CHANNELS].to_numpy(dtype=float)  # (T, 8)
        T = len(mat)
        if T < FS // 2:
            continue

        if window is None:
            starts = [0]
            seg_len = T
        else:
            seg_len = window
            if T < window:
                starts = [0]          # trial shorter than window -> use whole trial
                seg_len = T
            else:
                starts = list(range(0, T - window + 1, step))

        for s in starts:
            seg = mat[s:s + seg_len]
            X.append(segment_features(seg))
            y.append(int(labels[0]))
            groups.append(int(trial_id))

    return np.array(X), np.array(y), np.array(groups)


def fisher_scores(X, y):
    classes = np.unique(y)
    if len(classes) != 2:
        return None
    a = X[y == classes[0]]
    b = X[y == classes[1]]
    num = (a.mean(0) - b.mean(0)) ** 2
    den = a.var(0) + b.var(0) + 1e-12
    return num / den


def make_models():
    """The four ML models. Scaling matters for LDA/LogReg/SVM, not for RF."""
    return {
        "LDA":          make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "LogReg":       make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "SVM (RBF)":    make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }


def trials_per_class(y, groups):
    """Count distinct trials per class (for choosing a safe number of grouped folds)."""
    seen = {}
    for lbl, g in zip(y, groups):
        seen.setdefault(g, lbl)
    counts = {}
    for lbl in seen.values():
        counts[int(lbl)] = counts.get(int(lbl), 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser(description="Windowed CV separability (read-only).")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--classes", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--window", type=int, default=250, help="window length in samples (EEGNet uses 250)")
    ap.add_argument("--step", type=int, default=75, help="step between windows (EEGNet uses 75 = 70%% overlap)")
    ap.add_argument("--no-window", action="store_true", help="disable windowing (one vector per trial)")
    args = ap.parse_args()

    window = None if args.no_window else args.window
    step = args.step

    print("\n###### WINDOWED CROSS-VALIDATED SEPARABILITY (read-only) ######")
    print("Features: log band-power (mu 8-12, beta 13-30 Hz) for Ch1..Ch8")
    if window is None:
        print("Windowing: OFF (one feature vector per trial)")
    else:
        ov = (1 - step / window) * 100
        print(f"Windowing: {window}-sample window, {step}-sample step ({ov:.0f}% overlap) -- EEGNet-style")
    print("Models: LDA, LogReg, SVM (RBF), RandomForest. Chance = 50%.")
    print("Reported: GROUPED(by-trial, trustworthy) | LEAKY(window-shuffled, EEGNet-style/optimistic)\n")

    summary = {}
    for subj in args.subjects:
        data = load_subject(subj, args.classes)
        if data is None:
            print(f"SUBJECT {subj}: no CSVs found -- skipped.")
            continue

        X, y, groups = build_dataset(data, window=window, step=step)
        if len(X) == 0 or len(np.unique(y)) != 2:
            print(f"SUBJECT {subj}: insufficient/unbalanced data -- skipped.\n")
            continue

        n_trials = len(np.unique(groups))
        win_per_class = {int(c): int((y == c).sum()) for c in np.unique(y)}
        tpc = trials_per_class(y, groups)

        print("=" * 78)
        print(f"SUBJECT {subj}: {n_trials} trials -> {len(X)} windows | "
              f"windows/class={win_per_class} | trials/class={tpc}")

        # Grouped (leak-free) folds: cannot exceed min trials-per-class
        g_folds = max(2, min(args.folds, min(tpc.values())))
        if HAS_SGKF:
            grouped_cv = StratifiedGroupKFold(n_splits=g_folds, shuffle=True, random_state=42)
        else:
            grouped_cv = GroupKFold(n_splits=g_folds)

        # Leaky (window-shuffled) folds
        l_folds = max(2, min(args.folds, min(win_per_class.values())))
        leaky_cv = StratifiedKFold(n_splits=l_folds, shuffle=True, random_state=42)

        print(f"  {'Model':<14}{'GROUPED (by-trial)':<24}{'LEAKY (window-shuffled)'}")
        best_grouped = 0.0
        for name, model in make_models().items():
            try:
                g = cross_val_score(model, X, y, cv=grouped_cv, groups=groups, scoring="accuracy")
                g_str = f"{g.mean()*100:5.1f}% (+/-{g.std()*100:4.1f})"
                best_grouped = max(best_grouped, g.mean() * 100)
            except Exception as e:
                g_str = f"err: {e}"
            try:
                l = cross_val_score(model, X, y, cv=leaky_cv, scoring="accuracy")
                l_str = f"{l.mean()*100:5.1f}% (+/-{l.std()*100:4.1f})"
            except Exception as e:
                l_str = f"err: {e}"
            print(f"  {name:<14}{g_str:<24}{l_str}")

        fs = fisher_scores(X, y)
        if fs is not None:
            feat_names = [f"{ch}-{band}" for ch in EEG_CHANNELS for band in BANDS]
            order = np.argsort(fs)[::-1]
            top = ", ".join(f"{feat_names[i]}={fs[i]:.3f}" for i in order[:5])
            print(f"  Top discriminative features (Fisher): {top}")
        summary[subj] = best_grouped
        print()

    print("=" * 78)
    print("SUMMARY (best GROUPED/by-trial accuracy across the 4 models, chance = 50%)")
    for s, a in summary.items():
        verdict = "separable signal present" if a >= 62 else \
                  ("weak/borderline" if a >= 56 else "NOT separable (~chance)")
        print(f"  Subject {s}: {a:5.1f}%  -> {verdict}")
    print("\nNotes:")
    print("  * Trust the GROUPED column. If GROUPED ~50% but LEAKY is high, the model")
    print("    is only memorising within-trial windows -- not real generalization.")
    print("  * If GROUPED is high but EEGNet gives ~50%, the deep pipeline is the issue;")
    print("    if GROUPED ~50% too, the recording lacks separable signal (data/paradigm).\n")


if __name__ == "__main__":
    main()
