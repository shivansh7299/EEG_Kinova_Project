import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, filtfilt, resample_poly, resample, lfilter
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split
import pickle
import joblib
from sklearn.pipeline import Pipeline
from pathlib import Path
from sklearn.model_selection import train_test_split






class EEG_MI_Dataprocessor:
    def __init__(self, file=None, chunk_only = False):
        if not chunk_only:
            assert type(file) == str, f"Please provide a valid file path!, got :{file}"
            self.file = file
    def process_chunk(self, eeg_data):
        uniformed_trials = self.uniform_len(eeg_data, int(250 * 1))
        fs = 250
        lcut = 8
        hcut = 30
        order = 5
        filtered_trials = self.apply_filters(uniformed_trials, lcut, hcut, fs, order)
        # print(filtered_trials.shape)
        normalized_trials = self.zscore_normalize_trials(filtered_trials)
        # print(normalized_trials.shape)
        return normalized_trials
    
    def process_data(self):
        trials, labels = self.load_data_file(self.file)
        trials  = [i[int(250 * (3)):, :] for i in trials]
        assert len(trials) == len(labels), "n Trials and n Labels doesn't match"
        uniformed_trials = self.uniform_len(trials, int(250 * (3)))
        print(uniformed_trials.shape)
        fs = 250
        lcut = 8
        hcut = 30
        order = 5
        filtered_trials = self.apply_filters(uniformed_trials, lcut, hcut, fs, order)
        print(filtered_trials.shape)
        normalized_trials = self.zscore_normalize_trials(filtered_trials)
        print(normalized_trials.shape)
        return normalized_trials, np.array(labels)
    
    def zscore_normalize_trials(self, trials):
        """
        Apply z-score normalization per trial and channel.

        Parameters:
            trials (np.ndarray): EEG data, shape (n_trials, time_steps, n_channels)

        Returns:
            np.ndarray: Normalized EEG data of same shape
        """
        trials_norm = np.zeros_like(trials)
        for i in range(trials.shape[0]):
            for ch in range(trials.shape[2]):
                signal = trials[i, :, ch]
                mean = np.mean(signal)
                std = np.std(signal)
                trials_norm[i, :, ch] = (signal - mean) / (std + 1e-8)
        return trials_norm
    
    def uniform_len(self, trials, target_len=500):
        resampled = [resample(trial, target_len, axis=0) for trial in trials]
        return np.stack(resampled)
    
    def apply_filters(self, data, lowcut, highcut, fs, order=5):
        """
        Apply a Butterworth bandpass filter to 3D EEG data.

        Parameters:
            data (np.ndarray): EEG array of shape (n_trials, time_steps, n_channels)
            lowcut (float): Low cutoff frequency (Hz)
            highcut (float): High cutoff frequency (Hz)
            fs (int): Sampling frequency (Hz)
            order (int): Order of the Butterworth filter

        Returns:
            np.ndarray: Filtered EEG array of same shape
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        b, a = butter(order, [low, high], btype="band")

        filtered = np.zeros_like(data)

        for trial_idx in range(data.shape[0]):
            for ch in range(data.shape[2]):
                # Apply filter on time axis (axis=1)
                filtered[trial_idx, :, ch] = filtfilt(b, a, data[trial_idx, :, ch])

        return filtered

    def load_data_file(self, file_path):
        file_path = file_path
        colnames = [
            "timestamp",
            "task",
            "phase",
            "channel_1",
            "channel_2",
            "channel_3",
            "channel_4",
            "channel_5",
            "channel_6",
            "channel_7",
            "channel_8",
        ]
        data = pd.read_csv(file_path, header=None, usecols=range(11), names=colnames)
        data.sort_values("timestamp", ascending=True)
        return self.extract_eeg_trials_and_labels(data)

    def extract_eeg_trials_and_labels(self, data: pd.DataFrame):
        """
        From raw EEG DataFrame, extract valid trials and return separate lists:
        - EEG channel data arrays
        - Task labels (left, right, still)

        Parameters:
            data (pd.DataFrame): Original EEG dataframe with 'task', 'phase', and 8 channels.

        Returns:
            trials (List[np.ndarray]): List of EEG data arrays (shape: [time_steps, 8])
            labels (List[str]): Corresponding task label for each trial
        """
        # Detect task or phase change
        changes = (data["task"] != data["task"].shift()) | (
            data["phase"] != data["phase"].shift()
        )
        changed_rows = data[changes]

        trials_raw = []
        current_task = None
        current_phases = []
        start_idx = None
        expected_phases = ["baseline", "instruction", "reach"]
        change_indices = changed_rows.index.tolist()

        for i, idx in enumerate(change_indices):
            row = changed_rows.loc[idx]
            task = row["task"]
            phase = row["phase"]

            if current_task != task:
                current_task = task
                current_phases = []
                start_idx = None

            if phase == expected_phases[len(current_phases)]:
                current_phases.append(phase)
                if len(current_phases) == 1:
                    start_idx = idx

                if current_phases == expected_phases:
                    end_idx = (
                        change_indices[i + 1] if i + 1 < len(change_indices) else len(data)
                    )
                    trial_df = data.loc[start_idx : end_idx - 1]
                    trials_raw.append(trial_df)
                    current_task = None
                    current_phases = []
                    start_idx = None

        # Extract EEG and labels
        trials = []
        labels = []
        for trial in trials_raw:
            eeg_data = trial[[f"channel_{i}" for i in range(1, 9)]].values
            task_label = trial["task"].iloc[0]
            trials.append(eeg_data)
            labels.append(task_label)

        print(f"✅ Extracted {len(trials)} trials with labels.")
        return trials, labels
    




def split_eeg_data_with_counts(
    data, labels, val_size=0.2, test_size=0.2, random_state=42
):
    """
    Splits EEG data into train, validation, and test sets with stratified labels.
    Converts string labels to integers and prints class distribution and label mapping.

    Parameters:
        data (np.ndarray): EEG data of shape (n_trials, time, channels)
        labels (np.ndarray or list): Labels of shape (n_trials,)
        val_size (float): Fraction of training data to use as validation
        test_size (float): Fraction of total data to reserve for testing
        random_state (int): Random seed

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    labels = np.array(labels)

    # Step 0: Encode labels if they are strings
    if labels.dtype.kind in {"U", "S", "O"}:
        unique_classes = np.unique(labels)
        label_map = {label: idx for idx, label in enumerate(unique_classes)}
        labels_int = np.array([label_map[label] for label in labels])

        # Print mapping
        print("\n✅ Label Mapping:")
        for k, v in label_map.items():
            print(f"  {v}: {k}")
    else:
        labels_int = labels
        label_map = None

    # Step 1: Split into temp_train and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        data,
        labels_int,
        test_size=test_size,
        stratify=labels_int,
        random_state=random_state,
    )

    # Step 2: Split temp_train into train and val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )

    # --- Count per class ---
    def count_labels(y):
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))

    df = (
        pd.DataFrame(
            {
                "Class": sorted(set(np.concatenate([y_train, y_val, y_test]))),
                "Train": pd.Series(count_labels(y_train)),
                "Val": pd.Series(count_labels(y_val)),
                "Test": pd.Series(count_labels(y_test)),
            }
        )
        .fillna(0)
        .astype(int)
    )

    print("\n✅ Class Distribution:")
    print(df.to_string(index=False))

    return X_train, X_val, X_test, y_train, y_val, y_test

def window_eeg_with_labels(X, y, fs=250, win_sec=1.0, step_sec=1.0, drop_last=True):
    """
    X: (N, T, C) EEG array
    y: (N,) labels per trial
    fs: sampling rate (Hz)
    win_sec: window length in seconds
    step_sec: stride in seconds (use < win_sec for overlap)
    drop_last: if True, drop any trailing incomplete window

    Returns:
      Xw: (N_total_windows, win_samples, C)
      yw: (N_total_windows,)
      groups: (N_total_windows,) original trial index for GroupKFold
    """
    assert X.ndim == 3, "X must be (N, T, C)"
    assert y.ndim == 1 and y.shape[0] == X.shape[0], "y must be (N,) and align with X"

    N, T, C = X.shape
    win = int(round(win_sec * fs))
    step = int(round(step_sec * fs))
    if win > T:
        raise ValueError(f"Window ({win} samples) longer than trial length ({T}).")

    Xw_list, yw_list, grp_list = [], [], []

    for i in range(N):
        starts = range(0, T - win + 1, step)
        for s in starts:
            e = s + win
            if e <= T:
                Xw_list.append(X[i, s:e, :])
                yw_list.append(y[i])
                grp_list.append(i)
            elif not drop_last:
                # pad last short window if requested
                seg = X[i, s:T, :]
                pad = np.zeros((win - seg.shape[0], C), dtype=X.dtype)
                Xw_list.append(np.vstack([seg, pad]))
                yw_list.append(y[i])
                grp_list.append(i)

    Xw = np.stack(Xw_list, axis=0) if Xw_list else np.empty((0, win, C), dtype=X.dtype)
    yw = np.asarray(yw_list)
    groups = np.asarray(grp_list)
    return Xw, yw, groups

from .dataset import EEGDataset

def prepare_datasets(file=None, val_size = 0.2, test_size = 0.1, augment = False):
    data_processor = EEG_MI_Dataprocessor(file)
    eeg, labels = data_processor.process_data()
    # Windowing trials to 1sec segments
    windowed_eeg, windowed_labels, groups = window_eeg_with_labels(eeg, np.array(labels), win_sec=1, step_sec=1)
    X_train, X_val, X_test, y_train, y_val, y_test = split_eeg_data_with_counts(
    data=windowed_eeg,
    labels=windowed_labels,
    val_size=val_size,
    test_size=test_size
    )
    
    train_dataset = EEGDataset(X_train, y_train, augment=augment)
    val_dataset = EEGDataset(X_val, y_val, augment=False)
    test_dataset = EEGDataset(X_test, y_test, augment=False)
    
    return train_dataset, val_dataset, test_dataset