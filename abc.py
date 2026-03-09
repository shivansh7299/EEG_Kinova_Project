# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import mne
from glob import glob

# %%
# Load dataset
# data = pd.read_csv('data/shivansh_five_class_run_2.csv', header=None)
# print(data.head())

# Load and combine multiple files
# files = [
#     # 'data/EEG_bar_trials_20_classes_3_20251118_164048.csv',
#     # 'data/EEG_bar_trials_20_classes_3_20251118_174039.csv',
#     # 'data/EEG_bar_trials_20_classes_3_20251118_174039.csv',
#     'data/EEG_bar_trials_30_classes_3_20251118_161525.csv'
# ]

# data_list = [pd.read_csv(f) for f in files]
# data = pd.concat(data_list, ignore_index=True)
# print(f"Combined data shape: {data.shape}")

files = sorted(glob('data/EEG_bar_trials_*.csv'))
print(f"Found {len(files)} files:")
for f in files:
    print(f"  - {f}")

if len(files) == 0:
    print("WARNING: No files found! Check your data directory path.")
else:
    data_list = [pd.read_csv(f) for f in files]
    data = pd.concat(data_list, ignore_index=True)
    print(f"Combined data shape: {data.shape}")
    print(f"Unique trials: {data['trial_index'].nunique()}")
    print(f"Unique class labels: {data['class_label'].unique()}")

# %%
def ensure_fixed_length(arr, n_samples=2250, n_channels=8):
    """Ensure a 2D array has exactly n_samples rows and n_channels columns by truncating or padding with zeros.

    - If `arr` is empty, returns zeros of shape (n_samples, n_channels).
    - If `arr` has fewer/more columns than `n_channels`, columns are padded/truncated.
    - If `arr` has fewer rows than `n_samples`, rows are padded; if more, rows are truncated.
    """
    if arr is None or arr.size == 0:
        return np.zeros((n_samples, n_channels), dtype=float)

    rows, cols = arr.shape
    # fix columns
    if cols > n_channels:
        arr = arr[:, :n_channels]
    elif cols < n_channels:
        pad_cols = np.zeros((rows, n_channels - cols), dtype=arr.dtype)
        arr = np.hstack([arr, pad_cols])

    # fix rows
    if rows >= n_samples:
        return arr[:n_samples, :]
    else:
        pad = np.zeros((n_samples - rows, arr.shape[1]), dtype=arr.dtype)
        return np.vstack([arr, pad])


def trails_by_time(data, fs=250, eeg_col_start=5, eeg_col_end=13, n_samples=1000):
    """Build trials with fixed shape (n_samples, n_channels) for the MI (stimulus) phase.
    Returns: trials (N_trials, n_samples, n_channels), trial_labels (N_trials,), trial_splits (N_trials,)
    """
    trials = []
    trial_labels = []
    trial_splits = []
    unique_trials = data['trial_index'].unique()
    print(f'Found {len(unique_trials)} unique trials.')

    # determine EEG columns (keep iloc behavior: columns 5:13 -> typically 8 channels)
    eeg_cols = data.columns[eeg_col_start:eeg_col_end]
    n_channels = len(eeg_cols)

    for trial in unique_trials:
        print(f'Processing trial {trial}...')
        if trial != 0:
            trial_data = data[data['trial_index'] == trial]
            label = trial_data['class_label'].values[0] if 'class_label' in trial_data.columns else None
            split = trial_data['split'].values[0] if 'split' in trial_data.columns else None

            # select MI (stimulus) rows only and extract EEG columns
            trial_MI = trial_data[trial_data['phase'] == 'stimulus']
            eeg_matrix = trial_MI.loc[:, eeg_cols].values if len(trial_MI) > 0 else np.zeros((0, n_channels))

            # ensure exactly n_samples rows and n_channels columns (truncate or pad)
            fixed = ensure_fixed_length(eeg_matrix, n_samples=n_samples, n_channels=n_channels)

            trials.append(fixed)
            trial_labels.append(label)
            trial_splits.append(split)

    trials = np.array(trials)
    trial_labels = np.array(trial_labels)
    trial_splits = np.array(trial_splits)

    print(f'Built {len(trials)} trials -> each trial shape: {trials.shape[1:]}')
    # return trials as (N, n_samples, n_channels)
    return trials, trial_labels, trial_splits

# Run and build trials (explicitly set n_samples to 2250 to trim/pad every trial)
trials, trial_labels, trial_splits = trails_by_time(data, n_samples=1000)


# %%
# %%
print(f"Total trials: {len(trials)}")
print(f"Class distribution across all trials:")
unique, counts = np.unique(trial_labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c} trials")

# %%
# Create MI / baseline views (kept for reference)
MI = data[data['phase'] == 'stimulus']
baseline = data[data['phase'] == 'baseline']
print(f'MI Readings shape (rows in raw data): {MI.shape}')
print(f'Baseline Readings shape: {baseline.shape}')

# Show shapes of the built trials
print(f'Number of trials built: {len(trials)}')
if len(trials) > 0:
    print(f'Each trial shape (n_samples, n_channels): {trials.shape[1:]}')

# Convert per-trial data into a stacked continuous array so downstream cells that expect
# `readings`/`labels`/`split` still work without many changes.
# - readings: shape (n_trials * n_samples, n_channels)
# - labels:   repeated trial label per sample -> shape (n_trials * n_samples,)
# - split:    repeated trial split per sample
if len(trials) > 0:
    n_samples = trials.shape[1]
    n_channels = trials.shape[2]
    readings = trials.reshape(-1, n_channels)  # stack trials into continuous samples
    labels = np.repeat(trial_labels, n_samples)
    split = np.repeat(trial_splits, n_samples)

    print(f'Stacked readings shape: {readings.shape}')
    print(f'Stacked labels shape: {labels.shape}')
    print(f'Unique class labels (trials): {np.unique(trial_labels)}')
else:
    print('No trials were built; `readings` will fall back to using raw MI rows')

# If you want to continue using original MI-based readings instead, comment out the stacking above
# and uncomment the following two lines to use the row-wise MI values (not trimmed per-trial):
# readings = MI.iloc[:, 5:13].values
# labels = MI['class_label'].values


# %%
# %%
# Balance data per class to use all files equally
from collections import Counter

train_idx = np.where(split == 'train')[0]
test_idx  = np.where(split == 'test')[0]

eeg_train = readings[train_idx]
lbl_train = labels[train_idx]

eeg_test = readings[test_idx]
lbl_test = labels[test_idx]

print("\n=== BEFORE balancing ===")
print(f"Train samples: {eeg_train.shape[0]}")
print(f"Train label distribution: {np.unique(lbl_train, return_counts=True)}")
print(f"Test samples: {eeg_test.shape[0]}")
print(f"Test label distribution: {np.unique(lbl_test, return_counts=True)}")

# Find minimum per class
train_counts = Counter(lbl_train)
test_counts = Counter(lbl_test)

min_train_per_class = min(train_counts.values())
min_test_per_class = min(test_counts.values())

print(f"\nMinimum samples per class in train: {min_train_per_class}")
print(f"Minimum samples per class in test: {min_test_per_class}")

# Balance function
def balance_data(eeg_data, labels_data, min_samples_per_class):
    balanced_eeg = []
    balanced_labels = []
    
    unique_classes = np.unique(labels_data)
    for cls in unique_classes:
        cls_idx = np.where(labels_data == cls)[0]
        selected_idx = np.random.choice(cls_idx, size=min_samples_per_class, replace=False)
        balanced_eeg.append(eeg_data[selected_idx])
        balanced_labels.append(labels_data[selected_idx])
    
    return np.vstack(balanced_eeg), np.concatenate(balanced_labels)

# Apply balancing
eeg_train = balance_data(eeg_train, lbl_train, min_train_per_class)[0]
lbl_train = balance_data(readings[train_idx], labels[train_idx], min_train_per_class)[1]

eeg_test = balance_data(eeg_test, lbl_test, min_test_per_class)[0]
lbl_test = balance_data(readings[test_idx], labels[test_idx], min_test_per_class)[1]

print("\n=== AFTER balancing ===")
print(f"Train samples: {eeg_train.shape[0]}")
print(f"Train label distribution: {np.unique(lbl_train, return_counts=True)}")
print(f"Test samples: {eeg_test.shape[0]}")
print(f"Test label distribution: {np.unique(lbl_test, return_counts=True)}")

# %%
import matplotlib.pyplot as plt
# Show one trial (trial index 0), channel 0
trial_idx = 2
ch = 2
plt.figure(figsize=(10,3))
plt.plot(trials[trial_idx, :, ch], label=f'Trial {trial_idx} Ch{ch}')
plt.title(f'Trial {trial_idx} — Channel {ch}')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# PSD check for same trial/channel
from scipy.signal import welch
f, Pxx = welch(trials[trial_idx, :, ch], fs=250, nperseg=1024)
plt.semilogy(f, Pxx); plt.xlim(0,60); plt.title('PSD'); plt.grid(True); plt.show()

# %%
# readings = MI.iloc[:, 5:13].values
# labels = MI['class_label'].values
# split = MI['split'].values
# print(f'Readings shape: {readings.shape}, Labels shape: {labels}')
# left_hand_count = np.sum(labels == 'left_arm')
# upper_data = MI.iloc[labels == 1]
# upper_data_readings = upper_data.iloc[:, 5:13].values

# middle_data = MI.iloc[labels == 2]
# middle_data_readings = middle_data.iloc[:, 5:13].values

# lower_data = MI.iloc[labels == 3]
# lower_data_readings = lower_data.iloc[:, 5:13].values

# print(f'Number of upper data labels: {upper_data_readings}')
# print(f'Number of middle data labels: {middle_data_readings}')
# print(f'Number of lower data labels: {lower_data_readings}')



# print(labels)
# la = []
# ra = []
# rl = []
# ll = []
# s = []
# # assume you already did:
# readings = MI.iloc[:, 5:13].values   # shape (n_samples, n_channels)
# labels   = MI['class_label'].values      # shape (n_samples,)

# s = []
# la = []
# ra = []
# rl = []
# ll = []

# n = len(labels)
# for i in range(n):
#     lbl = labels[i]                      # <- single string label for row i
#     row = readings[i, :]                 # <- 1D array for that sample's 8 channels
#     if lbl == 1:
#         s.append(row)
#     elif lbl == 2:
#         la.append(row)
#     else:
#         ra.append(row)

# # convert to numpy arrays if needed
# s = np.array(s)
# la = np.array(la)
# ra = np.array(ra)
# rl = np.array(rl)
# ll = np.array(ll)

# print(f'label 1 samples: {len(s)}')
# print(f'label 2 samples: {len(la)}')
# print(f'label 3 samples: {len(ra)}')
# print(f'Unique labels: {np.unique(labels)}')
# print(f'Labels distribution:\n{pd.Series(labels).value_counts()}')
# print(readings[:5])

# %%
# X_train = readings[split == 'train']
# y_train = labels[split == 'train']
# X_test = readings[split == 'test']
# y_test = labels[split == 'test']
# print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')

# %%

def bandpass_filter(signal, lowcut=4, highcut=40, fs=250, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


# %%


# Verify shape: (n_samples, n_channels)
print("readings.shape:", readings.shape)

fs = 250  # sampling frequency (Hz) - keep this consistent with your data

filtered_readings = np.zeros_like(readings)
for ch_idx in range(readings.shape[1]):  # loop over 8 channels
    raw_signal = readings[:, ch_idx]
    filtered_readings[:, ch_idx] = bandpass_filter(raw_signal, lowcut=4, highcut=40, fs=fs, order=4)

print("filtered_readings shape:", filtered_readings.shape)

# %%
plt.figure(figsize=(12, 10))
for i in range(readings.shape[1]):
    plt.subplot(8, 1, i+1)
    plt.plot(filtered_readings[:4000, i], color='red', alpha=0.8, label='Filtered')
    plt.title(f"Channel {i}")
    plt.xticks([])
    plt.tight_layout()

plt.legend(['Raw', 'Filtered'], loc='upper right')
plt.suptitle("EEG Channels Before and After Band-pass Filtering", fontsize=14, y=1.02)
plt.show()



# %%
from scipy.signal import spectrogram

ch = 0
for i in range(readings.shape[1]):
    f, t, Sxx = spectrogram(readings[4000:5000, i], fs, nperseg=256, noverlap=128)
    plt.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
    plt.title(f"Spectrogram – Channel {i}")
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.ylim(0, 50)
    plt.colorbar(label='Power (dB)')
    plt.show()




# %%
import mne


fs = 250
# Use realistic EEG channel names if possible
ch_names = ['Fz','C3','Cz','C4','P3','Pz','P4','Oz']
ch_types = ['eeg'] * 8
info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)

print("filtered_readings shape:", filtered_readings.shape)
raw_mne = mne.io.RawArray(filtered_readings.T, info)

# ✅ Add montage to provide electrode positions
montage = mne.channels.make_standard_montage('standard_1020')
raw_mne.set_montage(montage)
# --- Run ICA ---
ica = mne.preprocessing.ICA(n_components=8, random_state=42, max_iter=500)
print("Fitting ICA...")
ica.fit(raw_mne)

# --- Visualize components safely ---
ica.plot_components()  # now works, shows scalp maps
# OR if you only want time-series of activations:
# ica.plot_sources(raw_mne)

# --- Optionally exclude components ---
ica.exclude = [0]  # change after visual inspection
raw_clean = ica.apply(raw_mne)

filtered_clean = raw_clean.get_data().T
print("filtered_clean shape:", filtered_clean.shape)

# --- Quick comparison ---
plt.figure(figsize=(12, 4))
plt.plot(filtered_readings[4250:5000, 0], label='Before ICA', alpha=0.6)
plt.plot(filtered_clean[4250:5000, 0], label='After ICA', color='red', alpha=0.8)
plt.title('EEG Channel 1 – Before vs After ICA Artifact Removal')
plt.xlabel('Samples')
plt.ylabel('Amplitude (µV)')
plt.legend()
plt.show()

# %%

scaler = StandardScaler()
eeg_norm = scaler.fit_transform(filtered_readings)  # shape: (n_samples, 8)

print("eeg_norm shape:", eeg_norm)

# %%
# label_encoder = LabelEncoder()
# labels_encoded = label_encoder.fit_transform(labels)
# print("labels_encoded shape:", labels_encoded)
# print(labels)

# Verify labels are numeric
print("Unique labels before encoding:", np.unique(labels))
print("Labels dtype:", labels.dtype)

# Since labels are already 1, 2, 3 format, just convert to int
labels_encoded = labels.astype(int)
print("labels_encoded shape:", labels_encoded.shape)
print("Unique encoded labels:", np.unique(labels_encoded))
print("Label distribution:", np.unique(labels_encoded, return_counts=True))

# %%
# # Split indices for train/test
# train_idx_filtered = np.where(split == 'train')[0]
# test_idx_filtered  = np.where(split == 'test')[0]

# eeg_train_data = eeg_norm[train_idx_filtered]
# lbl_train = labels_encoded[train_idx_filtered]

# eeg_test_data = eeg_norm[test_idx_filtered]
# lbl_test = labels_encoded[test_idx_filtered]

# print("Train samples:", eeg_train_data.shape[0])
# print("Test samples :", eeg_test_data.shape[0])
# print("Train label distribution:", np.unique(lbl_train, return_counts=True))
# print("Test label distribution:", np.unique(lbl_test, return_counts=True))

print("Train samples:", eeg_train.shape[0])
print("Test samples :", eeg_test.shape[0])

# %%
# -------------------------------
def create_epochs(eeg_array, labels_array, stride, window, amp_thresh=150):
    X, y = [], []
    for start in range(0, len(eeg_array) - window, stride):
        end = start + window
        epoch = eeg_array[start:end, :]
        epoch_labels = labels_array[start:end]
        # majority label in the window
        majority = np.bincount(epoch_labels).argmax()
        purity = np.mean(epoch_labels == majority)
        if purity >= 0.9 and np.max(np.abs(epoch)) < amp_thresh:
            X.append(epoch)
            y.append(majority)
    return np.array(X), np.array(y)

# %%
# train_idx = np.where(split == 'train')[0]
# test_idx  = np.where(split == 'test')[0]

# eeg_train, lbl_train = eeg_norm[train_idx], labels_encoded[train_idx]
# eeg_test,  lbl_test  = eeg_norm[test_idx],  labels_encoded[test_idx]
# print("Train samples:", eeg_train.shape[0])
# print("Test samples :", eeg_test.shape[0])

# %%
# window_sec = 1
# window_size = fs * window_sec          # 250 samples per 1 s
# train_stride = window_size // 2        # 50 % overlap
# test_stride  = window_size             # no overlap
# amp_thresh = 150   

# # X_train, y_train = create_epochs(eeg_train, lbl_train,
# #                                  stride=train_stride, window=window_size)
# # X_test, y_test = create_epochs(eeg_test, lbl_test,
# #                                stride=test_stride, window=window_size)

# X_train, y_train = create_epochs(eeg_train_data, lbl_train,
#                                  stride=train_stride, window=window_size)
# X_test, y_test = create_epochs(eeg_test_data, lbl_test,
#                                stride=test_stride, window=window_size)

# print("Train epochs:", X_train.shape, "Test epochs:", X_test.shape)

window_sec = 1
window_size = fs * window_sec          # 250 samples per 1 s
train_stride = window_size // 2        # 50% overlap
test_stride  = window_size             # no overlap
amp_thresh = 150   

X_train, y_train = create_epochs(eeg_norm_train, lbl_train,
                                 stride=train_stride, window=window_size)
X_test, y_test = create_epochs(eeg_norm_test, lbl_test,
                               stride=test_stride, window=window_size)

print("Train epochs:", X_train.shape, "Test epochs:", X_test.shape)
print("Train epoch labels:", np.unique(y_train, return_counts=True))
print("Test epoch labels:", np.unique(y_test, return_counts=True))

# %%
def normalize_epoch(epoch):
    mean = epoch.mean(axis=0, keepdims=True)
    std = epoch.std(axis=0, keepdims=True) + 1e-8
    return (epoch - mean) / std

X_train = np.array([normalize_epoch(e) for e in X_train])
X_test  = np.array([normalize_epoch(e) for e in X_test])

# --- Reshape for CTNet ---
X_train_ctnet = np.transpose(X_train, (0, 2, 1))  # (N, 8, 250)
X_test_ctnet  = np.transpose(X_test,  (0, 2, 1))


print("CTNet train shape:", X_train_ctnet.shape)
print("CTNet test shape:", X_test_ctnet.shape)

# %%
np.save("X_train_ctnet.npy", X_train_ctnet)
np.save("y_train_ctnet.npy", y_train)
np.save("X_test_ctnet.npy",  X_test_ctnet)
np.save("y_test_ctnet.npy",  y_test)

# %%



