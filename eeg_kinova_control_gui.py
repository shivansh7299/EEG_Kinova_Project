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
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QTextEdit, QProgressBar, QGroupBox,
    QSpinBox, QComboBox, QFileDialog, QMessageBox, QCheckBox, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Conda env Python for training / inference (PyTorch, model libs)
CONDA_PYTHON = PROJECT_ROOT / ".conda" / "python.exe"


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


# ============================================================================
# Tab: Training
# ============================================================================
class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        gb = QGroupBox("Training")
        gb_layout = QVBoxLayout()

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

        self.data_path = str(PROJECT_ROOT)
        gb.setLayout(gb_layout)
        layout.addWidget(gb)

        self.run_btn = QPushButton("Start Training")
        self.run_btn.setStyleSheet("font-size: 14px; padding: 10px; background-color: #FF9800; color: white;")
        self.run_btn.clicked.connect(self.run_training)
        layout.addWidget(self.run_btn)

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

    def run_training(self):
        if not CONDA_PYTHON.exists():
            QMessageBox.critical(
                self, "Conda Env Not Found",
                f"Conda Python not found at:\n{CONDA_PYTHON}\n\n"
                "Training requires the conda environment with PyTorch."
            )
            return

        conda_py = str(CONDA_PYTHON)
        tasks = []
        if self.train_eegnet.isChecked():
            tasks.append(("EEGNet", [conda_py, str(PROJECT_ROOT / "EEGNet_new_training.py")], self.data_path))
        if self.train_fbmsnet.isChecked():
            tasks.append(("FBMSNet", [conda_py, str(PROJECT_ROOT / "FBMSNet" / "train_custom_fbmsnet_updated.py")], str(PROJECT_ROOT / "FBMSNet")))
        if self.train_ctnet.isChecked():
            tasks.append(("CTNet", [conda_py, str(PROJECT_ROOT / "ctnet_training.py")], self.data_path))

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
        self._run_tasks(tasks, 0)

    def _run_tasks(self, tasks, idx):
        if idx >= len(tasks):
            self.log.append("\nAll training complete! Check model accuracy above.")
            self.run_btn.setEnabled(True)
            return

        name, cmd, cwd = tasks[idx]
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
        worker.output.connect(lambda t: self.log.append(t))
        worker.finished_signal.connect(
            lambda code, msg: self._on_train_done(code, msg, tasks, idx)
        )
        self.worker = worker
        worker.start()

    def _on_train_done(self, code, msg, tasks, idx):
        self.log.append(f"Training finished (exit code {code}).")
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
        row.addStretch()
        gb_layout.addLayout(row)

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

    def run_eeg_only(self):
        if not CONDA_PYTHON.exists():
            QMessageBox.critical(self, "Conda Env Not Found",
                                 f"Conda Python not found at:\n{CONDA_PYTHON}")
            return
        model_name = self.model_combo.currentText()
        self.log.append(f"Launching EEG-only Kinova control with {model_name}...")
        try:
            script = PROJECT_ROOT / "kinova_eeg_controller.py"
            subprocess.Popen(
                [str(CONDA_PYTHON), str(script), "--model", model_name],
                cwd=str(PROJECT_ROOT),
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
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
        gb_layout.addLayout(row)

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

    def run_eeg_opencv(self):
        if not CONDA_PYTHON.exists():
            QMessageBox.critical(self, "Conda Env Not Found",
                                 f"Conda Python not found at:\n{CONDA_PYTHON}")
            return
        model_name = self.model_combo.currentText()
        self.log.append(f"Launching EEG + OpenCV Kinova control with {model_name}...")
        try:
            script = PROJECT_ROOT / "kinova_eeg_opencv_controller.py"
            subprocess.Popen(
                [str(CONDA_PYTHON), str(script), "--model", model_name],
                cwd=str(PROJECT_ROOT),
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            self.log.append("EEG+OpenCV Kinova controller launched in new window.")
        except Exception as e:
            self.log.append(f"Error: {e}")
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
