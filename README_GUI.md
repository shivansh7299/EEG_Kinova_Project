# EEG-Kinova Control GUI

## Overview

The `eeg_kinova_control_gui.py` provides a unified interface for the full EEG-Kinova experiment pipeline:

1. **Data Collection** – Launch the EEG data collection GUI
2. **Preprocessing** – Run CTNet, FBMSNet, and EEGNet preprocessing notebooks
3. **Training** – Train EEGNet, FBMSNet, and CTNet models
4. **Option 5: EEG → Kinova** – Real-time EEG prediction drives the Kinova arm (no OpenCV)
5. **Option 6: EEG + OpenCV** – EEG prediction + OpenCV ball tracking; match = fast, mismatch = slow

## Quick Start

```bash
python eeg_kinova_control_gui.py
```

## Workflow

### 1. Data Collection
- Click **Start Data Collection** to open the data collection window
- Collect EEG data with the Unicorn device
- CSV files are saved in the project root (e.g. `EEG_bar_trials_20_classes_2_20251222_184334.csv`)

### 2. Preprocessing
- Select **2** or **3** classes
- Browse to the folder containing CSV files (or use default `data/`)
- Click **Run Preprocessing**
- CSVs are copied to `data/2_class/` or `data/3_class/`
- Notebooks run and produce `.npy` files in the project root

### 3. Training
- Select which models to train (EEGNet, FBMSNet, CTNet)
- Data directory should contain the `.npy` files (project root after preprocessing)
- Click **Start Training**
- Models are saved as `ctnet_best_model.pth`, `eegnet_best_model.pth`, `fbmsnet_best_model.pth`

### 4. Option 5: EEG → Kinova
- Select the model (EEGNet, FBMSNet, CTNet)
- Click **Start EEG → Kinova Control**
- Real-time EEG predictions move the arm: class 0 = left, class 1 = right

### 5. Option 6: EEG + OpenCV
- Select the model
- Click **Start EEG + OpenCV Kinova Control**
- Calibrate the camera (click TL, TR, BR, BL corners)
- If EEG prediction **matches** OpenCV direction → arm moves **fast**
- If EEG prediction **mismatches** → arm moves **slow**

## Files Created

| File | Purpose |
|------|---------|
| `eeg_kinova_control_gui.py` | Main GUI |
| `real_time_eeg_predictor.py` | Real-time EEG model inference |
| `kinova_eeg_controller.py` | Option 5: EEG-only arm control |
| `kinova_eeg_opencv_controller.py` | Option 6: EEG + OpenCV arm control |

## Dependencies

- PyQt5
- PyTorch
- NumPy, SciPy
- UnicornPy (EEG hardware)
- Kortex API (Kinova robot)
- OpenCV (Option 6)
- filterpy (Option 6)
- jupyter, nbconvert (preprocessing)

## Configuration

**Controller hardware settings:** Edit `hardware_config.py` for the real-time
controllers (Option 5, Option 6) and predictor. Values mirror:
- `OpenCV/track_ball_kinova_test_fixed.py` – Kinova arm
- `dataCollection.py` – EEG headset (250 Hz, 4 samples/read)
