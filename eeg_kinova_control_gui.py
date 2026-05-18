# ============================================================================
# EEG-Kinova Control GUI
# ============================================================================
# Orchestrates: Data Collection → Preprocessing → Training → Real-time Control
# Options: 1) Data Collection, 2) Preprocessing, 3) Training, 4) EEG-only Arm,
#          5) EEG + OpenCV (match=fast, mismatch=slow)
# ============================================================================

import sys
import os
import subprocess
import shutil
import glob
import re
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QTextEdit, QProgressBar, QGroupBox,
    QSpinBox, QComboBox, QFileDialog, QMessageBox, QCheckBox, QFrame, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Conda settings for training / inference (PyTorch, model libs)
# GUI can run on any Python; heavy scripts can be launched in a separate conda env.
CONDA_ENV_NAME = os.environ.get("EEG_KINOVA_CONDA_ENV", "torch37")


def _resolve_conda_launcher():
    """Return command prefix used to run Python scripts in the training env."""
    local_python = PROJECT_ROOT / ".conda" / "python.exe"
    if local_python.exists():
        return [str(local_python)], None

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return [conda_exe, "run", "--no-capture-output", "-n", CONDA_ENV_NAME, "python"], None

    common_conda = [
        Path(r"C:\ProgramData\Anaconda3\Scripts\conda.exe"),
        Path.home() / "anaconda3" / "Scripts" / "conda.exe",
        Path.home() / "miniconda3" / "Scripts" / "conda.exe",
    ]
    for candidate in common_conda:
        if candidate.exists():
            return [str(candidate), "run", "--no-capture-output", "-n", CONDA_ENV_NAME, "python"], None

    err = (
        "No training Python found.\n\n"
        "Expected one of:\n"
        f"1) Local env: {local_python}\n"
        "2) Conda executable in CONDA_EXE\n"
        "3) Standard Anaconda/Miniconda install path\n\n"
        f"Current target env name: {CONDA_ENV_NAME}\n"
        "Set env var EEG_KINOVA_CONDA_ENV if your env has a different name."
    )
    return None, err


def _launch_persistent_console(cmd):
    """Launch a command in a console that stays open after the process exits on Windows."""
    if sys.platform == 'win32':
        console_cmd = ["cmd", "/k", *cmd]
        creationflags = subprocess.CREATE_NEW_CONSOLE
    else:
        console_cmd = cmd
        creationflags = 0
    return subprocess.Popen(
        console_cmd,
        cwd=str(PROJECT_ROOT),
        creationflags=creationflags,
    )


# ============================================================================
# Background Worker Thread
# ============================================================================
class WorkerThread(QThread):
    """Runs a subprocess and emits output/status."""
    output = pyqtSignal(str)
    finished_signal = pyqtSignal(int, str)  # exit_code, message

    def __init__(self, cmd, cwd=None, env=None):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd or str(PROJECT_ROOT)
        self.env = env or os.environ.copy()
        self.process = None

    def run(self):
        try:
            self.process = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.output.emit(line.rstrip())
            self.process.wait()
            self.finished_signal.emit(self.process.returncode, "Process completed.")
        except Exception as e:
            self.output.emit(f"Error: {e}")
            self.finished_signal.emit(-1, str(e))

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()


# ============================================================================
# Notebook Execution Worker (for preprocessing)
# ============================================================================
class NotebookWorker(QThread):
    output = pyqtSignal(str)
    finished_signal = pyqtSignal(int, str)

    def __init__(self, notebook_path, n_class):
        super().__init__()
        self.notebook_path = Path(notebook_path)
        self.n_class = n_class

    def run(self):
        import json
        try:
            # Patch notebook to use correct data path (data/2_class or data/3_class)
            with open(self.notebook_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            data_pattern = f"data/{self.n_class}_class"
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    src = cell.get("source", [])
                    if isinstance(src, list):
                        new_src = []
                        for line in src:
                            # Replace data/2_class or data/3_class with our path
                            if "data/2_class" in line or "data/3_class" in line:
                                line = line.replace("data/2_class", data_pattern).replace("data/3_class", data_pattern)
                            new_src.append(line)
                        cell["source"] = new_src

            temp_nb = self.notebook_path.parent / f"_temp_{self.notebook_path.name}"
            with open(temp_nb, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1)

            cmd = [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=600",
                "--inplace",
                str(temp_nb)
            ]
            self.output.emit(f"Executing {self.notebook_path.name}...")
            env = os.environ.copy()
            env["MPLBACKEND"] = "Agg"
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            for line in iter(proc.stdout.readline, ''):
                if line:
                    self.output.emit(line.rstrip())
            proc.wait()

            # Copy output .npy files from temp run to project root (notebook runs in place)
            # The notebook saves to current dir, so outputs are in PROJECT_ROOT
            if temp_nb.exists():
                try:
                    temp_nb.unlink()
                except Exception:
                    pass
            self.finished_signal.emit(proc.returncode, f"Notebook {self.notebook_path.name} finished.")
        except Exception as e:
            self.output.emit(f"Error: {e}")
            self.finished_signal.emit(-1, str(e))


# ============================================================================
# Tab: Data Collection
# ============================================================================
class DataCollectionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        gb = QGroupBox("Data Collection Settings")
        gb_layout = QVBoxLayout()

        row0 = QHBoxLayout()
        row0.addWidget(QLabel("Subject Number:"))
        self.subject_no = QSpinBox()
        self.subject_no.setRange(1, 999)
        self.subject_no.setValue(1)
        row0.addWidget(self.subject_no)
        row0.addStretch()
        gb_layout.addLayout(row0)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Classes:"))
        self.num_classes = QSpinBox()
        self.num_classes.setRange(2, 5)
        self.num_classes.setValue(2)
        row1.addWidget(self.num_classes)
        row1.addWidget(QLabel("Trials per class:"))
        self.trials_per_class = QSpinBox()
        self.trials_per_class.setRange(1, 200)
        self.trials_per_class.setValue(20)
        row1.addWidget(self.trials_per_class)
        row1.addStretch()
        gb_layout.addLayout(row1)
        gb.setLayout(gb_layout)
        layout.addWidget(gb)

        self.run_btn = QPushButton("Start Data Collection")
        self.run_btn.setStyleSheet("font-size: 14px; padding: 10px; background-color: #4CAF50; color: white;")
        self.run_btn.clicked.connect(self.run_data_collection)
        layout.addWidget(self.run_btn)

        layout.addWidget(QLabel("Progress / Log:"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(200)
        layout.addWidget(self.log)

        self.setLayout(layout)

    def run_data_collection(self):
        """Launch dataCollection.py in a subprocess (it has its own PyQt GUI)."""
        self.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Launching Data Collection GUI...")
        script = PROJECT_ROOT / "dataCollection.py"
        if not script.exists():
            self.log.append(f"Error: {script} not found!")
            QMessageBox.critical(self, "Error", f"dataCollection.py not found at:\n{script}")
            return
        subj = self.subject_no.value()
        n_classes = self.num_classes.value()
        n_trials = self.trials_per_class.value()
        output_dir = PROJECT_ROOT / "data" / f"Subject {subj}" / f"{n_classes}_class"
        cmd = [
            sys.executable, str(script),
            "--subject", str(subj),
            "--classes", str(n_classes),
            "--trials", str(n_trials),
        ]
        self.log.append(f"Subject: {subj} | Classes: {n_classes} | Trials: {n_trials}")
        self.log.append(f"CSV will be saved to: {output_dir}")
        try:
            subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
            self.log.append("Data collection window opened. Collect data, then proceed to Preprocessing tab.")
        except Exception as e:
            self.log.append(f"Error: {e}")
            QMessageBox.critical(self, "Error", str(e))


# ============================================================================
# Tab: Preprocessing
# ============================================================================
class PreprocessingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.workers = []
        self.current_subject_class_dir = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        gb = QGroupBox("Preprocessing Settings")
        gb_layout = QVBoxLayout()

        row0 = QHBoxLayout()
        row0.addWidget(QLabel("Subject Number:"))
        self.subject_no = QSpinBox()
        self.subject_no.setRange(1, 20)
        self.subject_no.setValue(1)
        self.subject_no.valueChanged.connect(self._update_data_path_label)
        row0.addWidget(self.subject_no)
        row0.addStretch()
        gb_layout.addLayout(row0)

        row1 = QHBoxLayout() 
        row1.addWidget(QLabel("Number of classes (2 or 3):"))
        self.num_classes = QSpinBox()
        self.num_classes.setRange(2, 3)
        self.num_classes.setValue(2)
        self.num_classes.valueChanged.connect(self._update_data_path_label)
        row1.addWidget(self.num_classes)
        row1.addStretch()
        gb_layout.addLayout(row1)

        self.data_path_label = QLabel("")
        self.data_path_label.setStyleSheet("color: #aaa; font-size: 11px;")
        gb_layout.addWidget(self.data_path_label)
        self._update_data_path_label()

        gb.setLayout(gb_layout)
        layout.addWidget(gb)

        self.run_btn = QPushButton("Run Preprocessing (CTNet, FBMSNet, EEGNet)")
        self.run_btn.setStyleSheet("font-size: 14px; padding: 10px; background-color: #2196F3; color: white;")
        self.run_btn.clicked.connect(self.run_preprocessing)
        layout.addWidget(self.run_btn)

        layout.addWidget(QLabel("Progress / Log:"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(250)
        layout.addWidget(self.log)

        self.setLayout(layout)

    def _update_data_path_label(self):
        subj = self.subject_no.value()
        n_class = self.num_classes.value()
        path = PROJECT_ROOT / "data" / f"Subject {subj}" / f"{n_class}_class"
        self.data_path_label.setText(f"Data source: {path}")

    def run_preprocessing(self):
        subj = self.subject_no.value()
        n_class = self.num_classes.value()
        subject_class_dir = PROJECT_ROOT / "data" / f"Subject {subj}" / f"{n_class}_class"
        self.current_subject_class_dir = subject_class_dir

        if not subject_class_dir.exists():
            QMessageBox.warning(
                self, "No Data",
                f"Subject folder not found:\n{subject_class_dir}\n\n"
                "Run Data Collection for this subject first."
            )
            return

        csv_files = list(subject_class_dir.glob("EEG_*.csv"))
        if not csv_files:
            QMessageBox.warning(
                self, "No Data",
                f"No EEG CSV files found in:\n{subject_class_dir}\n\n"
                "Run Data Collection for this subject first."
            )
            return

        # Copy CSVs from subject folder to data/N_class/ (notebooks expect this path)
        class_dir = PROJECT_ROOT / "data" / f"{n_class}_class"
        class_dir.mkdir(parents=True, exist_ok=True)
        for f in csv_files:
            try:
                shutil.copy2(f, class_dir / f.name)
                self.log.append(f"Copied {f.name} → data/{n_class}_class/")
            except Exception as e:
                self.log.append(f"Copy warning: {e}")

        notebooks = [
            PROJECT_ROOT / "preprocessing_3_class_CTNet.ipynb",
            PROJECT_ROOT / "preprocessing_3_class_FBMSNet.ipynb",
            PROJECT_ROOT / "preprocessing_3_class_EEGNet.ipynb",
        ]
        for nb in notebooks:
            if not nb.exists():
                self.log.append(f"Warning: {nb.name} not found, skipping.")

        self.run_btn.setEnabled(False)
        self.log.append(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting preprocessing for Subject {subj}, {n_class}-class data...")
        self.log.append(f"Source: {subject_class_dir}  ({len(csv_files)} CSV files)")

        self._run_next_notebook(notebooks, 0, n_class)

    def _run_next_notebook(self, notebooks, idx, n_class):
        if idx >= len(notebooks):
            self._stage_subject_npy_outputs()
            self.log.append("\nAll preprocessing complete!")
            self.run_btn.setEnabled(True)
            return

        nb = notebooks[idx]
        if not nb.exists():
            self._run_next_notebook(notebooks, idx + 1, n_class)
            return

        worker = NotebookWorker(str(nb), n_class)
        worker.output.connect(lambda t: self.log.append(t))
        worker.finished_signal.connect(
            lambda code, msg: self._on_notebook_done(code, msg, notebooks, idx, n_class)
        )
        self.workers.append(worker)
        worker.start()

    def _on_notebook_done(self, code, msg, notebooks, idx, n_class):
        self.log.append(f"Exit code {code}: {msg}")
        if code != 0:
            self.log.append("Preprocessing may have failed. Check log above.")
        self._run_next_notebook(notebooks, idx + 1, n_class)

    def _stage_subject_npy_outputs(self):
        """Copy generated preprocessing .npy files to selected subject folder for training tab use."""
        if self.current_subject_class_dir is None:
            return

        self.current_subject_class_dir.mkdir(parents=True, exist_ok=True)
        npy_files = [
            "X_train_eegnet.npy", "y_train_eegnet.npy", "X_test_eegnet.npy", "y_test_eegnet.npy",
            "X_train_ctnet.npy", "y_train_ctnet.npy", "X_test_ctnet.npy", "y_test_ctnet.npy",
            "X_train_fbmsnet.npy", "y_train_fbmsnet.npy", "X_test_fbmsnet.npy", "y_test_fbmsnet.npy",
        ]

        copied = 0
        for fname in npy_files:
            src = PROJECT_ROOT / fname
            if not src.exists():
                continue
            dst = self.current_subject_class_dir / fname
            try:
                shutil.copy2(src, dst)
                copied += 1
                self.log.append(f"Staged {fname} → {dst}")
            except Exception as e:
                self.log.append(f"Stage warning for {fname}: {e}")

        if copied == 0:
            self.log.append("No .npy preprocessing artifacts found in project root to stage.")
        else:
            self.log.append(f"Staged {copied} .npy files to subject folder: {self.current_subject_class_dir}")


# ============================================================================
# Tab: Training
# ============================================================================
class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.total_tasks = 0
        self.current_task_index = 0
        self.current_task_name = ""
        self.current_epoch_total = 0
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        gb = QGroupBox("Training")
        gb_layout = QVBoxLayout()

        row0 = QHBoxLayout()
        row0.addWidget(QLabel("Subject Number:"))
        self.subject_no = QSpinBox()
        self.subject_no.setRange(1, 999)
        self.subject_no.setValue(1)
        self.subject_no.valueChanged.connect(self._update_data_dir_for_subject)
        row0.addWidget(self.subject_no)
        row0.addWidget(QLabel("Classes:"))
        self.num_classes = QSpinBox()
        self.num_classes.setRange(2, 5)
        self.num_classes.setValue(2)
        self.num_classes.valueChanged.connect(self._update_data_dir_for_subject)
        row0.addWidget(self.num_classes)
        row0.addStretch()
        gb_layout.addLayout(row0)

        self.train_eegnet = QCheckBox("Train EEGNet")
        self.train_eegnet.setChecked(True)
        self.train_fbmsnet = QCheckBox("Train FBMSNet")
        self.train_fbmsnet.setChecked(True)
        self.train_ctnet = QCheckBox("Train CTNet")
        self.train_ctnet.setChecked(True)

        gb_layout.addWidget(self.train_eegnet)
        gb_layout.addWidget(self.train_fbmsnet)
        gb_layout.addWidget(self.train_ctnet)

        row = QHBoxLayout()
        row.addWidget(QLabel("Data directory (with .npy files):"))
        self.data_dir = QLabel(str(PROJECT_ROOT))
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_data)
        row.addWidget(self.data_dir)
        row.addWidget(self.browse_btn)
        gb_layout.addLayout(row)

        self.data_path = str(self._subject_data_dir())
        self.data_dir.setText(self.data_path)
        self.data_hint = QLabel("")
        self.data_hint.setStyleSheet("color: #aaa; font-size: 11px;")
        gb_layout.addWidget(self.data_hint)
        self._update_data_dir_for_subject()
        gb.setLayout(gb_layout)
        layout.addWidget(gb)

        self.run_btn = QPushButton("Start Training")
        self.run_btn.setStyleSheet("font-size: 14px; padding: 10px; background-color: #FF9800; color: white;")
        self.run_btn.clicked.connect(self.run_training)
        layout.addWidget(self.run_btn)

        self.overall_progress_label = QLabel("Overall progress")
        layout.addWidget(self.overall_progress_label)
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setValue(0)
        self.overall_progress.setFormat("%p%")
        layout.addWidget(self.overall_progress)

        self.epoch_progress_label = QLabel("Current model epoch progress")
        layout.addWidget(self.epoch_progress_label)
        self.epoch_progress = QProgressBar()
        self.epoch_progress.setRange(0, 100)
        self.epoch_progress.setValue(0)
        self.epoch_progress.setFormat("Waiting for epoch logs...")
        layout.addWidget(self.epoch_progress)

        layout.addWidget(QLabel("Progress / Log:"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(300)
        layout.addWidget(self.log)

        self.setLayout(layout)

    def browse_data(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Directory", self.data_path)
        if folder:
            self.data_path = folder
            self.data_dir.setText(folder)
            self.data_hint.setText("Using manually selected data directory.")

    def _subject_data_dir(self):
        return PROJECT_ROOT / "data" / f"Subject {self.subject_no.value()}" / f"{self.num_classes.value()}_class"

    def _update_data_dir_for_subject(self):
        path = self._subject_data_dir()
        self.data_path = str(path)
        self.data_dir.setText(self.data_path)
        self.data_hint.setText(f"Auto path from subject/class: {path}")

    def run_training(self):
        launcher, err = _resolve_conda_launcher()
        if launcher is None:
            QMessageBox.critical(
                self, "Conda Env Not Found",
                err
            )
            return

        subject_id = self.subject_no.value()
        tasks = []
        if self.train_eegnet.isChecked():
            tasks.append(("EEGNet", launcher + [str(PROJECT_ROOT / "EEGNet_new_training.py"), "--data-dir", self.data_path, "--subject-id", str(subject_id)], self.data_path))
        if self.train_fbmsnet.isChecked():
            tasks.append(("FBMSNet", launcher + [str(PROJECT_ROOT / "FBMSNet" / "train_custom_fbmsnet_updated.py"), "--data-dir", self.data_path, "--subject-id", str(subject_id)], self.data_path))
        if self.train_ctnet.isChecked():
            tasks.append(("CTNet", launcher + [str(PROJECT_ROOT / "ctnet_training.py"), "--data-dir", self.data_path, "--subject-id", str(subject_id)], self.data_path))

        if not tasks:
            QMessageBox.warning(self, "Warning", "Select at least one model to train.")
            return

        # Check for EEGNet/CTNet data (required for at least one selected model)
        data_dir = Path(self.data_path)
        eegnet_ok = (data_dir / "X_train_eegnet.npy").exists()
        ctnet_ok = (data_dir / "X_train_ctnet.npy").exists()
        if self.train_eegnet.isChecked() and not eegnet_ok:
            QMessageBox.warning(
                self, "Missing Data",
                "X_train_eegnet.npy not found. Run Preprocessing first (Tab 2)."
            )
            return
        if self.train_ctnet.isChecked() and not ctnet_ok:
            QMessageBox.warning(
                self, "Missing Data",
                "X_train_ctnet.npy not found. Run Preprocessing first (Tab 2)."
            )
            return

        self.run_btn.setEnabled(False)
        self.total_tasks = len(tasks)
        self.current_task_index = 0
        self.current_task_name = ""
        self.current_epoch_total = 0
        self.overall_progress.setValue(0)
        self.epoch_progress.setRange(0, 100)
        self.epoch_progress.setValue(0)
        self.epoch_progress.setFormat("Waiting for epoch logs...")
        self._run_tasks(tasks, 0)

    def _update_overall_progress(self, completed_tasks):
        if self.total_tasks <= 0:
            self.overall_progress.setValue(0)
            return
        pct = int((completed_tasks / self.total_tasks) * 100)
        self.overall_progress.setValue(max(0, min(100, pct)))

    def _set_epoch_total(self, total):
        if total <= 0:
            return
        self.current_epoch_total = total
        self.epoch_progress.setRange(0, total)
        self.epoch_progress.setValue(0)
        self.epoch_progress.setFormat("%v/%m epochs")

    def _update_epoch_progress(self, current, total=None):
        if total is not None and total > 0 and total != self.current_epoch_total:
            self._set_epoch_total(total)
        if self.current_epoch_total <= 0:
            return
        current = max(0, min(current, self.current_epoch_total))
        self.epoch_progress.setValue(current)

    def _parse_progress_from_line(self, line):
        # Patterns like: "Epoch [3/250]" or "Epoch 003/300"
        m = re.search(r"Epoch\s*\[?\s*(\d+)\s*/\s*(\d+)\s*\]?", line, re.IGNORECASE)
        if m:
            self._update_epoch_progress(int(m.group(1)), int(m.group(2)))
            return

        # Pattern like: "Training for 300 epochs"
        m = re.search(r"training\s+for\s+(\d+)\s+epochs", line, re.IGNORECASE)
        if m:
            self._set_epoch_total(int(m.group(1)))
            return

        # tqdm-like fallback line with "Epochs" and "x/y"
        if "epoch" in line.lower():
            m = re.search(r"(\d+)\s*/\s*(\d+)", line)
            if m:
                cur = int(m.group(1))
                total = int(m.group(2))
                if 1 <= cur <= total <= 5000:
                    self._update_epoch_progress(cur, total)

    def _on_worker_output(self, text):
        self.log.append(text)
        self._parse_progress_from_line(text)

    def _run_tasks(self, tasks, idx):
        if idx >= len(tasks):
            self.log.append("\nAll training complete! Check model accuracy above.")
            self._update_overall_progress(self.total_tasks)
            if self.current_epoch_total > 0:
                self.epoch_progress.setValue(self.current_epoch_total)
            self.run_btn.setEnabled(True)
            return

        name, cmd, cwd = tasks[idx]
        self.current_task_index = idx
        self.current_task_name = name
        self.current_epoch_total = 0
        self._update_overall_progress(idx)
        self.overall_progress_label.setText(f"Overall progress: model {idx + 1}/{len(tasks)} ({name})")
        self.epoch_progress_label.setText(f"{name} epoch progress")
        # Show busy state immediately so the GUI doesn't look frozen while waiting for first epoch logs.
        self.epoch_progress.setRange(0, 0)
        self.epoch_progress.setFormat("Running... waiting for first epoch output")
        self.log.append(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting {name} training...")

        env = os.environ.copy()
        # Prevent matplotlib from blocking (plt.show() opens window and hangs in subprocess)
        env["MPLBACKEND"] = "Agg"
        env["PYTHONUNBUFFERED"] = "1"

        # FBMSNet expects data in FBMSNet/data/ - preprocessing saves to project root
        # Copy npy files to FBMSNet/data/ if needed
        if name == "FBMSNet":
            fbmsnet_data = PROJECT_ROOT / "FBMSNet" / "data"
            fbmsnet_data.mkdir(parents=True, exist_ok=True)
            required = ["X_train_fbmsnet.npy", "y_train_fbmsnet.npy", "X_test_fbmsnet.npy", "y_test_fbmsnet.npy"]
            missing = [f for f in required if not (PROJECT_ROOT / f).exists()]
            if missing:
                self.log.append(f"Skipping FBMSNet: missing {', '.join(missing)}. Run Preprocessing first.")
                self._run_tasks(tasks, idx + 1)
                return
            for f in required:
                src = PROJECT_ROOT / f
                dst = fbmsnet_data / f
                if src.exists():
                    shutil.copy2(src, dst)
                    self.log.append(f"Copied {f} to FBMSNet/data/")

        worker = WorkerThread(cmd, cwd=cwd, env=env)
        worker.output.connect(self._on_worker_output)
        worker.finished_signal.connect(
            lambda code, msg: self._on_train_done(code, msg, tasks, idx)
        )
        self.worker = worker
        worker.start()

    def _on_train_done(self, code, msg, tasks, idx):
        self.log.append(f"Training finished (exit code {code}).")
        if self.current_epoch_total > 0:
            self.epoch_progress.setValue(self.current_epoch_total)
        self._update_overall_progress(idx + 1)
        self._run_tasks(tasks, idx + 1)


# ============================================================================
# Tab: Option 5 - EEG Only (Kinova arm from EEG predictions)
# ============================================================================
class EEGOnlyTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        gb = QGroupBox("Option 5: EEG → Kinova Arm (No OpenCV)")
        gb_layout = QVBoxLayout()
        gb_layout.addWidget(QLabel("Use the highest-accuracy model to predict real-time EEG and move the Kinova arm."))

        row = QHBoxLayout()
        row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["EEGNet", "FBMSNet", "CTNet"])
        row.addWidget(self.model_combo)
        row.addWidget(QLabel("Subject:"))
        self.subject_no = QSpinBox()
        self.subject_no.setRange(1, 999)
        self.subject_no.setValue(1)
        row.addWidget(self.subject_no)
        row.addWidget(QLabel("Classes:"))
        self.num_classes = QSpinBox()
        self.num_classes.setRange(2, 5)
        self.num_classes.setValue(2)
        row.addWidget(self.num_classes)
        row.addStretch()
        gb_layout.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Model checkpoint (.pth):"))
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Optional: choose from subject results/checkpoints")
        row2.addWidget(self.model_path_input)
        self.pick_latest_btn = QPushButton("Use Latest Subject Model")
        self.pick_latest_btn.clicked.connect(self.pick_latest_subject_model)
        row2.addWidget(self.pick_latest_btn)
        self.browse_model_btn = QPushButton("Browse")
        self.browse_model_btn.clicked.connect(self.browse_model_file)
        row2.addWidget(self.browse_model_btn)
        gb_layout.addLayout(row2)

        self.enable_stabilization = QCheckBox("Enable 20s stabilization before prediction")
        self.enable_stabilization.setChecked(False)
        gb_layout.addWidget(self.enable_stabilization)

        gb.setLayout(gb_layout)
        layout.addWidget(gb)

        self.run_btn = QPushButton("Start EEG → Kinova Control")
        self.run_btn.setStyleSheet("font-size: 14px; padding: 10px; background-color: #9C27B0; color: white;")
        self.run_btn.clicked.connect(self.run_eeg_only)
        layout.addWidget(self.run_btn)

        layout.addWidget(QLabel("Log:"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(150)
        layout.addWidget(self.log)

        self.setLayout(layout)

    def browse_model_file(self):
        start_dir = PROJECT_ROOT / "results"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select model checkpoint",
            str(start_dir),
            "PyTorch model (*.pth);;All files (*)",
        )
        if file_path:
            self.model_path_input.setText(file_path)

    def pick_latest_subject_model(self):
        model_name = self.model_combo.currentText()
        subject_dir = PROJECT_ROOT / "results" / f"Subject_{self.subject_no.value():02d}" / f"{self.num_classes.value()}_class" / model_name
        if not subject_dir.exists():
            QMessageBox.warning(self, "Not Found", f"Subject model folder not found:\n{subject_dir}")
            return

        run_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name.startswith("run_")], key=lambda p: p.stat().st_mtime, reverse=True)
        for run_dir in run_dirs:
            checkpoints_dir = run_dir / "checkpoints"
            if not checkpoints_dir.exists():
                continue
            candidates = sorted(checkpoints_dir.glob("*best*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                candidates = sorted(checkpoints_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                self.model_path_input.setText(str(candidates[0]))
                self.log.append(f"Selected latest subject model: {candidates[0]}")
                return

        QMessageBox.warning(self, "Not Found", f"No .pth model found in:\n{subject_dir}")

    def run_eeg_only(self):
        launcher, err = _resolve_conda_launcher()
        if launcher is None:
            QMessageBox.critical(self, "Conda Env Not Found", err)
            return
        model_name = self.model_combo.currentText()
        model_path = self.model_path_input.text().strip()
        stabilization_enabled = self.enable_stabilization.isChecked()
        self.log.append(f"Launching EEG-only Kinova control with {model_name}...")
        try:
            script = PROJECT_ROOT / "kinova_eeg_controller.py"
            cmd = launcher + [str(script), "--model", model_name]
            if model_path:
                cmd += ["--model-path", model_path]
            if stabilization_enabled:
                cmd += ["--stabilization-seconds", "20"]
            _launch_persistent_console(cmd)
            if stabilization_enabled:
                self.log.append("20s stabilization enabled. Robot will hold position before EEG prediction starts.")
            self.log.append("EEG-Kinova controller launched in new window.")
        except Exception as e:
            self.log.append(f"Error: {e}")
            QMessageBox.critical(self, "Error", str(e))


# ============================================================================
# Tab: Option 6 - EEG + OpenCV (match=fast, mismatch=slow)
# ============================================================================
class EEGOpenCVTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        gb = QGroupBox("Option 6: EEG + OpenCV (Match = Fast, Mismatch = Slow)")
        gb_layout = QVBoxLayout()
        gb_layout.addWidget(QLabel(
            "EEG model predicts direction. OpenCV tracks ball/bar. "
            "If prediction matches OpenCV → arm moves FAST. If mismatch → arm moves SLOW."
        ))

        row = QHBoxLayout()
        row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["EEGNet", "FBMSNet", "CTNet"])
        row.addWidget(self.model_combo)
        row.addWidget(QLabel("Subject:"))
        self.subject_no = QSpinBox()
        self.subject_no.setRange(1, 999)
        self.subject_no.setValue(1)
        row.addWidget(self.subject_no)
        row.addWidget(QLabel("Classes:"))
        self.num_classes = QSpinBox()
        self.num_classes.setRange(2, 5)
        self.num_classes.setValue(2)
        row.addWidget(self.num_classes)
        gb_layout.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Model checkpoint (.pth):"))
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Optional: choose from subject results/checkpoints")
        row2.addWidget(self.model_path_input)
        self.pick_latest_btn = QPushButton("Use Latest Subject Model")
        self.pick_latest_btn.clicked.connect(self.pick_latest_subject_model)
        row2.addWidget(self.pick_latest_btn)
        self.browse_model_btn = QPushButton("Browse")
        self.browse_model_btn.clicked.connect(self.browse_model_file)
        row2.addWidget(self.browse_model_btn)
        gb_layout.addLayout(row2)

        self.enable_stabilization = QCheckBox("Enable 20s stabilization before prediction")
        self.enable_stabilization.setChecked(False)
        gb_layout.addWidget(self.enable_stabilization)

        gb.setLayout(gb_layout)
        layout.addWidget(gb)

        self.run_btn = QPushButton("Start EEG + OpenCV Kinova Control")
        self.run_btn.setStyleSheet("font-size: 14px; padding: 10px; background-color: #E91E63; color: white;")
        self.run_btn.clicked.connect(self.run_eeg_opencv)
        layout.addWidget(self.run_btn)

        layout.addWidget(QLabel("Log:"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(150)
        layout.addWidget(self.log)

        self.setLayout(layout)

    def browse_model_file(self):
        start_dir = PROJECT_ROOT / "results"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select model checkpoint",
            str(start_dir),
            "PyTorch model (*.pth);;All files (*)",
        )
        if file_path:
            self.model_path_input.setText(file_path)

    def pick_latest_subject_model(self):
        model_name = self.model_combo.currentText()
        subject_dir = PROJECT_ROOT / "results" / f"Subject_{self.subject_no.value():02d}" / f"{self.num_classes.value()}_class" / model_name
        if not subject_dir.exists():
            QMessageBox.warning(self, "Not Found", f"Subject model folder not found:\n{subject_dir}")
            return

        run_dirs = sorted([p for p in subject_dir.iterdir() if p.is_dir() and p.name.startswith("run_")], key=lambda p: p.stat().st_mtime, reverse=True)
        for run_dir in run_dirs:
            checkpoints_dir = run_dir / "checkpoints"
            if not checkpoints_dir.exists():
                continue
            candidates = sorted(checkpoints_dir.glob("*best*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                candidates = sorted(checkpoints_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                self.model_path_input.setText(str(candidates[0]))
                self.log.append(f"Selected latest subject model: {candidates[0]}")
                return

        QMessageBox.warning(self, "Not Found", f"No .pth model found in:\n{subject_dir}")

    def run_eeg_opencv(self):
        launcher, err = _resolve_conda_launcher()
        if launcher is None:
            QMessageBox.critical(self, "Conda Env Not Found", err)
            return
        model_name = self.model_combo.currentText()
        model_path = self.model_path_input.text().strip()
        stabilization_enabled = self.enable_stabilization.isChecked()
        self.log.append(f"Launching EEG + OpenCV Kinova control with {model_name}...")
        self.log.append(f"[DEBUG] Launcher: {launcher}")
        try:
            script = PROJECT_ROOT / "kinova_eeg_opencv_controller.py"
            self.log.append(f"[DEBUG] Script path: {script}")
            self.log.append(f"[DEBUG] Script exists: {script.exists()}")
            cmd = launcher + [str(script), "--model", model_name]
            if model_path:
                cmd += ["--model-path", model_path]
            if stabilization_enabled:
                cmd += ["--stabilization-seconds", "20"]
            self.log.append(f"[DEBUG] Full command: {' '.join(cmd)}")
            print(f"\n[GUI DEBUG] Running command: {' '.join(cmd)}\n", file=sys.stderr)
            _launch_persistent_console(cmd)
            if stabilization_enabled:
                self.log.append("20s stabilization enabled. Robot will hold position before EEG prediction starts.")
            self.log.append("EEG+OpenCV Kinova controller launched in new window.")
        except Exception as e:
            self.log.append(f"Error: {e}")
            print(f"\n[GUI ERROR] {e}\n", file=sys.stderr)
            QMessageBox.critical(self, "Error", str(e))


# ============================================================================
# Main Window
# ============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG-Kinova Control Center")
        self.setMinimumSize(800, 700)
        self.resize(900, 750)

        central = QWidget()
        layout = QVBoxLayout()

        title = QLabel("EEG-Kinova Experiment Control")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.tabs = QTabWidget()
        self.tabs.addTab(DataCollectionTab(), "1. Data Collection")
        self.tabs.addTab(PreprocessingTab(), "2. Preprocessing")
        self.tabs.addTab(TrainingTab(), "3. Training")
        self.tabs.addTab(EEGOnlyTab(), "5. EEG → Kinova")
        self.tabs.addTab(EEGOpenCVTab(), "6. EEG + OpenCV")
        layout.addWidget(self.tabs)

        central.setLayout(layout)
        self.setCentralWidget(central)

        # Dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #2b2b2b; color: #e0e0e0; }
            QGroupBox { font-weight: bold; border: 1px solid #555; border-radius: 5px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QTabWidget::pane { border: 1px solid #555; background-color: #333; }
            QTabBar::tab { background-color: #444; color: #fff; padding: 8px 16px; margin-right: 2px; }
            QTabBar::tab:selected { background-color: #2196F3; }
            QTextEdit { background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas; font-size: 12px; }
            QPushButton { background-color: #555; color: white; border: none; border-radius: 4px; }
            QPushButton:hover { background-color: #666; }
            QPushButton:disabled { background-color: #333; color: #888; }
            QSpinBox, QComboBox { background-color: #333; color: #fff; border: 1px solid #555; padding: 4px; }
        """)


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
