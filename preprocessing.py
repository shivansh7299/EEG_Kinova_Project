import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import spkit as sp
from datetime import datetime

CSV_PATH = r"C:\Users\Shivansh Sharma\Desktop\EEG_Kinova_Project\data\shivansh_five_class_run_2.csv"

# ---------------------------
# I/O & parsing configuration
# ---------------------------
HAS_HEADER = False          # set False if your CSV has no header row
TIME_FORMAT = "auto"
FS_FALLBACK = 256.0 


FILTER_BY_THIRD_COL = True
FILTER_VALUE = "mi"
LABEL = ['left_leg','still']    
RETIME_AFTER_FILTER = False

ATAR_MODE = "soft"          # 'soft' | 'linAtten' | 'elim'
BETA = 0.10
WAVELET = "db3"
USE_JOBLIB = True

USE_HIGHPASS = True
HIGHPASS_CUTOFF = 0.5  # Hz

PLOT_SECONDS = None
STACK_OFFSET = None

if HAS_HEADER:
    df = pd.read_csv(CSV_PATH)
else:
    df = pd.read_csv(CSV_PATH, header=None)

print(f"Loaded CSV shape (before filtering): {df.shape}")

if FILTER_BY_THIRD_COL:
    col3 = df.iloc[:, 2].astype(str).str.strip().str.lower()
    mask = (col3 == FILTER_VALUE.lower())
    kept = int(mask.sum())
    if kept == 0:
        raise ValueError(
            f'No rows matched 3rd column == "{FILTER_VALUE}". '
            f'Check the CSV and the filter value.'
        )
    df = df.loc[mask].reset_index(drop=True)
    print(f"Filtered rows where 3rd column == '{FILTER_VALUE}': kept {kept} rows. New shape: {df.shape}")

time_col = df.iloc[:, 0].astype(str).to_numpy()

def to_seconds(arr):
    try:
        return arr.astype(float)
    except Exception:
        pass
    try:
        t0 = datetime.strptime(arr[0], "%H:%M:%S.%f")
        base_sec = t0.hour*3600 + t0.minute*60 + t0.second + t0.microsecond/1e6
        out = []
        for a in arr:
            t = datetime.strptime(a, "%H:%M:%S.%f")
            sec = t.hour*3600 + t.minute*60 + t.second + t.microsecond/1e6
            out.append(sec - base_sec)
        return np.array(out)
    except Exception:
        raise ValueError("Timestamp column not float or HH:MM:SS.sss format")

if TIME_FORMAT == "seconds":
    t_sec = time_col.astype(float)
elif TIME_FORMAT == "datetime":
    ts = pd.to_datetime(time_col, utc=True)
    t_sec = (ts.view("int64") - ts.iloc[0].value) / 1e9
else:
    t_sec = to_seconds(time_col)

if RETIME_AFTER_FILTER:
    dt_tmp = np.diff(t_sec)
    dt_tmp = dt_tmp[~np.isnan(dt_tmp)]
    fs_est_tmp = None
    if len(dt_tmp) > 0 and np.all(dt_tmp > 0):
        fs_est_tmp = 1.0 / np.median(dt_tmp)
        if not (10 <= fs_est_tmp <= 2048):
            fs_est_tmp = None
    FS_tmp = float(fs_est_tmp) if fs_est_tmp is not None else FS_FALLBACK
    t_sec = np.arange(len(t_sec)) / FS_tmp
    print(f"[INFO] RETIME_AFTER_FILTER=True → rebuilt contiguous time with FS≈{FS_tmp:.2f} Hz")


dt = np.diff(t_sec)
dt = dt[~np.isnan(dt)]
fs_est = None
if len(dt) > 0 and np.all(dt > 0):
    fs_est = 1.0 / np.median(dt)
    if not (10 <= fs_est <= 2048):
        fs_est = None

FS = float(fs_est) if fs_est is not None else FS_FALLBACK
# print(f"Sampling rate FS = {FS:.2f} Hz (inferred: {fs_est is not None})")

X = df.iloc[:, 4:12].to_numpy(dtype=float)
n, ch = X.shape

try:
    time_series = pd.to_datetime(df.iloc[:, 0], format="%H:%M:%S.%f")
    dts = (time_series.diff().dt.total_seconds()).dropna()
    if len(dts) > 0:
        print("Median time step (s):", dts.median())
        print("Approx sampling rate (Hz):", 1 / dts.median())
except Exception:
    pass

print(f"Data shape after filter: samples={n}, channels={ch}")

if HAS_HEADER:
    ch_names = list(df.columns[4:4+ch])
else:
    ch_names = [f"Ch{i+1}" for i in range(ch)]

# Optional trimming for quick plotting
if PLOT_SECONDS is not None:
    n_keep = int(PLOT_SECONDS * FS)
    n_keep = min(n_keep, n)
    X = X[:n_keep, :]
    t_sec = t_sec[:n_keep]
    n = n_keep
    print(f"Trimmed to first {PLOT_SECONDS}s -> {n} samples")


if USE_HIGHPASS:
    Xf = sp.filter_X(X, band=[HIGHPASS_CUTOFF], btype='highpass', fs=FS, verbose=0)
else:
    Xf = X.copy()


XR = sp.eeg.ATAR(
    Xf.copy(),
    verbose=0,
    beta=BETA,
    OptMode=ATAR_MODE,
    wv=WAVELET,
    use_joblib=USE_JOBLIB,
)
assert XR.shape == Xf.shape
print(f"ATAR cleaned shape: {XR.shape}")



out = pd.DataFrame(np.column_stack([t_sec, XR]), columns=[df.columns[0]] + ch_names)
# If you need the original 3rd column retained in the output as well, you can add it:
# out = pd.concat([df.iloc[:, [0, 2]].reset_index(drop=True), pd.DataFrame(XR, columns=ch_names)], axis=1)
Y = out.iloc[:, 1:1+ch].to_numpy(dtype=float)  
n_clean, ch2 = Y.shape
print(f"Cleaned matrix shape saved: samples={n_clean}, channels={ch2}")
out_path = "eeg_cleaned_atar_MI_only.csv"
out.to_csv(out_path, index=False)
print(f"Saved cleaned CSV (MI only): {out_path}")

