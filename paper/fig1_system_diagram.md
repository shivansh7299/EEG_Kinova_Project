# Fig. 1 — System Architecture (IEEE BSN Paper)

Use this Mermaid diagram for slides, thesis, or to redraw in PowerPoint/Draw.io.
The LaTeX/TikZ version for the paper is in `fig1_system_architecture.tex`.

```mermaid
flowchart LR
    subgraph Sensing["Wearable Sensing"]
        U["Unicorn Hybrid Black<br/>8-ch EEG @ 250 Hz"]
    end

    subgraph Offline["Offline Pipeline (GUI)"]
        DC["Data Collection<br/>1s baseline → 2s cue → 5s MI"]
        PP["Preprocessing<br/>4–40 Hz BP, trial split, z-score"]
        TR["Training<br/>EEGNet / CTNet / FBMSNet"]
        CKPT["Subject-specific<br/>.pth checkpoints"]
    end

    subgraph RealTime["Real-Time Control"]
        BUF["Rolling buffer<br/>1 s window"]
        INF["Inference @ 2 Hz<br/>every 0.5 s"]
        ARM["Kinova Gen3<br/>left / right pose"]
    end

    subgraph Hybrid["Optional Fusion"]
        CV["OpenCV bar tracking<br/>match = fast, mismatch = slow"]
    end

    GUI["eeg_kinova_control_gui.py<br/>Collect → Preprocess → Train → Deploy"]

    U --> DC
    DC --> PP --> TR --> CKPT
    GUI -.-> DC
    GUI -.-> PP
    GUI -.-> TR

    U --> BUF
    CKPT --> INF
    BUF --> INF --> ARM
    CV --> INF
```

## Component map (code files)

| Block | File |
|---|---|
| Data collection | `dataCollection.py` |
| Preprocessing | `preprocessing_3_class_EEGNet.ipynb`, `_CTNet.ipynb`, `_FBMSNet.ipynb` |
| Training | `EEGNet_new_training.py`, `ctnet_training.py`, `FBMSNet/train_custom_fbmsnet_updated.py` |
| Real-time inference | `real_time_eeg_predictor.py` |
| EEG-only control | `kinova_eeg_controller.py` |
| EEG + vision hybrid | `kinova_eeg_opencv_controller.py` |
| Unified GUI | `eeg_kinova_control_gui.py` |

## Timing summary

| Parameter | Value |
|---|---|
| Sampling rate | 250 Hz |
| MI phase (offline) | 5 s (1250 samples) |
| Real-time window | 1 s (250 samples) |
| Prediction rate | 0.5 s (2 Hz) |
| Trials per class | 20 |
