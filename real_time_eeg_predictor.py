# ============================================================================
# Real-Time EEG Predictor
# ============================================================================
# Loads trained model and predicts from streaming EEG buffer.
# Supports EEGNet, CTNet. FBMSNet requires multiband preprocessing (9, 8, 248).
# ============================================================================

import os
import sys
import numpy as np
import torch
from pathlib import Path
from scipy.signal import butter, filtfilt

PROJECT_ROOT = Path(__file__).resolve().parent

# EEG params (match training)
from hardware_config import EEG_FS
FS = EEG_FS
N_CHANNELS = 8
WINDOW_SAMPLES = 500  # 2 second #250 for 1 sec
LOWCUT = 4
HIGHCUT = 40
FILTER_ORDER = 5


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut=LOWCUT, highcut=HIGHCUT, fs=FS, order=FILTER_ORDER):
    """Apply bandpass filter to EEG data. data: (samples, channels)"""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    filtered = np.zeros_like(data)
    for c in range(data.shape[1]):
        filtered[:, c] = filtfilt(b, a, data[:, c].astype(np.float64))
    return filtered.astype(np.float32)


def zscore_normalize(x, mean=None, std=None):
    """Normalize along last axis. If mean/std None, compute from data."""
    if mean is None:
        mean = np.mean(x, axis=-1, keepdims=True)
    if std is None:
        std = np.std(x, axis=-1, keepdims=True) + 1e-8
    return (x - mean) / std


class RealTimeEEGPredictor:
    """
    Real-time EEG predictor. Buffers samples, preprocesses, runs model inference.
    """

    def __init__(self, model_name="CTNet", model_path=None, device=None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.num_classes = 2
        self.window_samples = WINDOW_SAMPLES
        self.buffer = []
        self.model_path = model_path or self._find_model_path()

        self._load_model()

    def _find_model_path(self):
        """Find best model checkpoint for the given model type."""
        candidates = []
        if self.model_name == "EEGNet":
            candidates = [
                PROJECT_ROOT / "outputs" / "eegnet_new_best.pth",
                PROJECT_ROOT / "eegnet_best_model.pth",
            ]
        elif self.model_name == "CTNet":
            candidates = [
                PROJECT_ROOT / "ctnet_best_model.pth",
            ]
        elif self.model_name == "FBMSNet":
            candidates = [
                PROJECT_ROOT / "FBMSNet" / "fbmsnet_best_model.pth",
            ]
        for p in candidates:
            if p.exists():
                return str(p)
        raise FileNotFoundError(f"No {self.model_name} model found. Train first. Looked in: {candidates}")

    def _infer_eegnet_window_samples(self, state):
        """Infer EEG window length from checkpoint classifier shape when available."""
        from EEGNet_new_model import EEGNet

        target_in_features = None
        for k, v in state.items():
            if k.endswith("classify.weight") and hasattr(v, "shape") and len(v.shape) == 2:
                target_in_features = int(v.shape[1])
                break

        if target_in_features is None:
            return WINDOW_SAMPLES

        candidates = [WINDOW_SAMPLES, 250, 500, 750, 1000]
        seen = set()
        for samples in candidates:
            if samples in seen:
                continue
            seen.add(samples)
            try:
                probe = EEGNet(num_classes=self.num_classes, num_channels=N_CHANNELS, num_samples=samples)
                if probe.classify.weight.shape[1] == target_in_features:
                    return samples
            except Exception:
                continue

        return WINDOW_SAMPLES

    def _load_model(self):
        if self.model_name == "EEGNet":
            from EEGNet_new_model import EEGNet
            ckpt = torch.load(self.model_path, map_location=self.device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
                self.num_classes = ckpt.get("num_classes", 2)
            else:
                state = ckpt
                self.num_classes = 2

            self.window_samples = self._infer_eegnet_window_samples(state)
            self.model = EEGNet(num_classes=self.num_classes, num_channels=N_CHANNELS,
                               num_samples=self.window_samples).to(self.device)
            self.model.load_state_dict(state, strict=False)
        elif self.model_name == "CTNet":
            from ctnet_model import CTNetLite
            ckpt = torch.load(self.model_path, map_location=self.device)
            if isinstance(ckpt, dict):
                state = ckpt.get("model_state_dict", ckpt)
                self.num_classes = ckpt.get("num_classes")
            else:
                state = ckpt
                self.num_classes = None
            if self.num_classes is None:
                for k, v in state.items():
                    if "classifier" in k and "weight" in k:
                        self.num_classes = v.shape[0]
                        break
                self.num_classes = self.num_classes or 2
            self.window_samples = WINDOW_SAMPLES
            self.model = CTNetLite(n_channels=N_CHANNELS, n_timepoints=WINDOW_SAMPLES,
                                   n_classes=self.num_classes).to(self.device)
            self.model.load_state_dict(state, strict=False)
        elif self.model_name == "FBMSNet":
            sys.path.insert(0, str(PROJECT_ROOT / "FBMSNet"))
            from codes.centralRepo.networks import FBMSNet
            ckpt = torch.load(self.model_path, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)
            self.num_classes = ckpt.get("num_classes", 2)
            self.window_samples = WINDOW_SAMPLES
            # FBMSNet expects (batch, 9, 8, 248) - multiband
            self.model = FBMSNet(nChan=8, nTime=248, nClass=self.num_classes,
                                 temporalLayer='LogVarLayer', num_Feat=36,
                                 dilatability=8, dropoutP=0.6).to(self.device)
            self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def add_samples(self, samples):
        """
        Add new samples to buffer. samples: (N, channels) - use first 8 channels.
        """
        if samples.ndim == 1:
            samples = samples.reshape(-1, samples.size // N_CHANNELS)
        if samples.shape[1] > N_CHANNELS:
            samples = samples[:, :N_CHANNELS]
        for row in samples:
            self.buffer.append(row.tolist())
        # Keep only last WINDOW_SAMPLES
        if len(self.buffer) > self.window_samples:
            self.buffer = self.buffer[-self.window_samples:]

    def predict(self):
        """
        Run prediction on current buffer. Returns (class_idx, probs) or (None, None) if insufficient data.
        """
        if len(self.buffer) < self.window_samples:
            return None, None

        window = np.array(self.buffer[-self.window_samples:], dtype=np.float32)

        # Preprocess
        filtered = bandpass_filter(window)
        normalized = zscore_normalize(filtered)

        if self.model_name == "FBMSNet":
            # FBMSNet needs multiband: (9, 8, 248) - simplified: replicate to 9 bands
            x = normalized.T  # (8, 250)
            x = x[:, :248]  # trim to 248
            x = np.tile(x[np.newaxis, :, :], (9, 1, 1))  # (9, 8, 248)
            x_t = torch.FloatTensor(x).unsqueeze(0).to(self.device)
        elif self.model_name == "EEGNet":
            # (1, 1, 8, 250)
            x = normalized.T  # (8, 250)
            x_t = torch.FloatTensor(x).unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            # CTNet: (1, 8, 250)
            x = normalized.T  # (8, 250)
            x_t = torch.FloatTensor(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x_t)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))

        return pred, probs

    def clear_buffer(self):
        self.buffer = []


if __name__ == "__main__":
    # Quick test
    pred = RealTimeEEGPredictor("CTNet")
    # Add random data
    pred.add_samples(np.random.randn(250, 8).astype(np.float32))
    p, probs = pred.predict()
    print(f"Prediction: {p}, Probs: {probs}")
