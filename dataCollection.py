import sys, os, random, csv, threading, time, argparse
from pathlib import Path
from datetime import datetime
from enum import Enum
from PyQt5.QtCore import QTimer, Qt, QRect, QPoint
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QPushButton,
                              QProgressBar, QHBoxLayout, QSpinBox, QGroupBox)
from PyQt5.QtGui import QColor, QPalette, QPainter, QPolygon
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

try:
    import UnicornPy
    USE_UNICORN = True
except ImportError:
    UnicornPy = None
    USE_UNICORN = False
    print("UnicornPy not found - running in mock mode")


# ========== DISPLAY MODE SYSTEM ==========
class DisplayMode(Enum):
    """Enum to define different visual display modes"""
    BAR = "bar"
    SHAPES = "shapes"


class ColorDisplay(QWidget):
    """Base class for color display widgets"""
    def __init__(self, class_colors):
        super().__init__()
        self.class_colors = class_colors
        self.highlight_index = None
        self.setMinimumWidth(80)
        self.setMaximumWidth(120)
    
    def paintEvent(self, event):
        """Each display mode implements its own painting logic"""
        raise NotImplementedError("Subclasses must implement paintEvent()")
    
    def highlight_color(self, class_index):
        """Highlight one color (during stimulus)"""
        self.highlight_index = class_index
        self.update()
    
    def reset_colors(self):
        """Show all class colors (during instruction)"""
        self.highlight_index = None
        self.update()


class ColorBar(ColorDisplay):
    """Original horizontal bar display"""
    def paintEvent(self, event):
        painter = QPainter(self)
        bar_width = self.width()
        bar_height = self.height() // len(self.class_colors)

        for i, (cls, color) in enumerate(self.class_colors.items()):
            rect = QRect(0, i * bar_height, bar_width, bar_height)
            if self.highlight_index is None:
                painter.fillRect(rect, color)
            elif cls == self.highlight_index:
                painter.fillRect(rect, color)
            else:
                painter.fillRect(rect, QColor(30, 30, 30))
        painter.end()


class ColorShapes(ColorDisplay):
    """Display colored geometric shapes (circle, square, triangle, diamond, pentagon)"""
    def __init__(self, class_colors):
        super().__init__(class_colors)
        self.setMinimumWidth(150)
        self.setMaximumWidth(200)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        total_shapes = len(self.class_colors)
        shape_height = self.height() // total_shapes
        center_x = self.width() // 2
        shape_size = min(shape_height - 20, self.width() - 20)
        
        # Define shape drawing functions
        shape_types = [
            self._draw_circle,
            self._draw_square,
            self._draw_triangle,
            self._draw_diamond,
            self._draw_pentagon
        ]
        
        for i, (cls, color) in enumerate(self.class_colors.items()):
            center_y = i * shape_height + shape_height // 2
            
            # Determine if this shape should be highlighted
            if self.highlight_index is None:
                brush_color = color
            elif cls == self.highlight_index:
                brush_color = color
            else:
                brush_color = QColor(30, 30, 30)
            
            painter.setBrush(brush_color)
            painter.setPen(Qt.NoPen)
            
            # Draw the appropriate shape for this class
            shape_func = shape_types[i % len(shape_types)]
            shape_func(painter, center_x, center_y, shape_size)
        
        painter.end()
    
    def _draw_circle(self, painter, x, y, size):
        """Draw a circle"""
        painter.drawEllipse(QPoint(x, y), size // 2, size // 2)
    
    def _draw_square(self, painter, x, y, size):
        """Draw a square"""
        half = size // 2
        rect = QRect(x - half, y - half, size, size)
        painter.drawRect(rect)
    
    def _draw_triangle(self, painter, x, y, size):
        """Draw an upward-pointing triangle"""
        half = size // 2
        points = [
            QPoint(x, y - half),
            QPoint(x - half, y + half),
            QPoint(x + half, y + half)
        ]
        painter.drawPolygon(QPolygon(points))
    
    def _draw_diamond(self, painter, x, y, size):
        """Draw a diamond (rotated square)"""
        half = size // 2
        points = [
            QPoint(x, y - half),
            QPoint(x + half, y),
            QPoint(x, y + half),
            QPoint(x - half, y)
        ]
        painter.drawPolygon(QPolygon(points))
    
    def _draw_pentagon(self, painter, x, y, size):
        """Draw a regular pentagon"""
        import math
        radius = size // 2
        points = []
        for i in range(5):
            angle = math.radians(i * 72 - 90)
            px = x + radius * math.cos(angle)
            py = y + radius * math.sin(angle)
            points.append(QPoint(int(px), int(py)))
        painter.drawPolygon(QPolygon(points))


def create_color_display(display_mode, class_colors):
    """Factory function to create the appropriate display widget"""
    if display_mode == DisplayMode.BAR or display_mode == "bar":
        return ColorBar(class_colors)
    elif display_mode == DisplayMode.SHAPES or display_mode == "shapes":
        return ColorShapes(class_colors)
    else:
        raise ValueError(f"Unknown display mode: {display_mode}")


# ========== EEG RECORDING THREADS ==========
class UnicornRecorder(threading.Thread):
    def __init__(self, gui_ref):
        super().__init__(daemon=True)
        self.gui = gui_ref
        self.running = False
        self.recording = False  # NEW: separate flag for actual data recording
        self.device = None
        self.fs = 250
        self.num_channels = None

    def connect(self):
        devices = UnicornPy.GetAvailableDevices(True)
        if not devices:
            raise RuntimeError("No Unicorn device found.")
        
        print(f"Connecting to device: {devices[0]}")
        self.device = UnicornPy.Unicorn(devices[0])
        
        self.num_channels = self.device.GetNumberOfAcquiredChannels()
        print(f"Device has {self.num_channels} channels")
        print(f"Sample rate: {self.fs} Hz")
        
        try:
            self.device.StartAcquisition(False)
            print("Acquisition started successfully.")
        except Exception as e:
            print("Failed to start acquisition:", e)
            raise
    
    def start_recording(self):
        """Enable data recording (called after stabilization)"""
        self.recording = True
        print("Data recording ENABLED")
        
    def run(self):
        if self.device is None:
            print("No device connected.")
            return

        self.running = True
        
        samples_per_read = 4
        buffer_size = samples_per_read * self.num_channels * 4
        
        print(f"Thread running: {self.num_channels} channels, {samples_per_read} samples per read")
        print(f"Buffer size: {buffer_size} bytes")
        print("Data recording PAUSED (waiting for stabilization to complete)")

        total_samples = 0
        discarded_samples = 0
        last_flush_time = time.time()
        consecutive_read_errors = 0
        max_consecutive_errors = 40
        
        time.sleep(0.5)
        
        try:
            while self.running:
                receive_buffer = bytearray(buffer_size)
                try:
                    self.device.GetData(samples_per_read, receive_buffer, buffer_size)
                    consecutive_read_errors = 0
                except Exception as e:
                    consecutive_read_errors += 1
                    print(
                        f"EEG read warning ({consecutive_read_errors}/{max_consecutive_errors}): {e}"
                    )

                    # Try to recover from transient Bluetooth/USB hiccups.
                    if consecutive_read_errors >= max_consecutive_errors:
                        print("Too many consecutive read errors. Restarting acquisition...")
                        try:
                            self.device.StopAcquisition()
                        except Exception:
                            pass
                        try:
                            self.device.StartAcquisition(False)
                            consecutive_read_errors = 0
                            print("Acquisition restart successful.")
                        except Exception as restart_err:
                            print(f"Acquisition restart failed: {restart_err}")
                            raise

                    time.sleep(0.02)
                    continue

                data = np.frombuffer(receive_buffer, dtype=np.float32)
                data = data.reshape((samples_per_read, self.num_channels))
                
                for sample in data:
                    # Only record data if recording flag is enabled
                    if self.recording:
                        total_samples += 1
                        trial = self.gui.current_trial
                        cls = self.gui.current_class
                        phase = getattr(self.gui, "phase", "idle")
                        ts = datetime.now().strftime("%H:%M:%S.%f")
                        row = [trial, cls, phase, ts] + sample.tolist()
                        self.gui.csv_writer.writerow(row)
                    else:
                        discarded_samples += 1
                
                current_time = time.time()
                if current_time - last_flush_time >= 1.0:
                    if self.recording:
                        self.gui.log_file.flush()
                        print(f"✓ {total_samples} samples recorded ({total_samples/250:.1f}s)")
                    last_flush_time = current_time
                
        except KeyboardInterrupt:
            print("Recording interrupted by user")
        except Exception as e:
            print(f"Fatal error after {total_samples} samples: {e}")
            import traceback
            traceback.print_exc()

        print(f"EEG recording stopped. Recorded: {total_samples} samples ({total_samples/250:.1f}s)")
        print(f"Discarded during stabilization: {discarded_samples} samples")

    def stop(self):
        print("Stopping recorder thread...")
        self.running = False
        time.sleep(0.5)
        if self.device:
            try:
                self.device.StopAcquisition()
                print("Acquisition stopped.")
            except Exception as ex:
                print(f"Stop error: {ex}")
            try:
                del self.device
                self.device = None
                print("Device disconnected.")
            except Exception as ex:
                print(f"Disconnect error: {ex}")


class MockEEGRecorder(threading.Thread):
    def __init__(self, gui_ref):
        super().__init__(daemon=True)
        self.gui = gui_ref
        self.running = False
        self.recording = False  # NEW: separate flag for actual data recording
        self.num_channels = 17

    def start_recording(self):
        """Enable data recording (called after stabilization)"""
        self.recording = True
        print("Mock data recording ENABLED")

    def run(self):
        self.running = True
        print("Running mock EEG recorder (17 channels)")
        print("Data recording PAUSED (waiting for stabilization to complete)")
        
        while self.running:
            # Only record data if recording flag is enabled
            if self.recording:
                fake_data = [random.uniform(-100, 100) for _ in range(self.num_channels)]
                ts = datetime.now().strftime("%H:%M:%S.%f")
                trial = self.gui.current_trial
                cls = self.gui.current_class
                phase = getattr(self.gui, "phase", "idle")
                self.gui.csv_writer.writerow([trial, cls, phase, ts] + fake_data)
                self.gui.log_file.flush()
            time.sleep(1 / 250)

    def stop(self):
        self.running = False


# ========== MAIN EEG GUI ==========
class EEGTrialGUI(QWidget):
    def __init__(self, num_classes=5, trials_per_class=3, baseline_ms=3000, 
                 instruction_display_ms=3000, stim_ms=3000, stabilization_ms=20000, 
                 display_mode="bar", subject_no=1):
        super().__init__()

        self.num_classes = num_classes
        self.trials_per_class = trials_per_class
        self.baseline_ms = baseline_ms
        self.instruction_display_ms = instruction_display_ms
        self.stim_ms = stim_ms
        self.stabilization_ms = stabilization_ms
        self.subject_no = subject_no

        self.display_mode = display_mode

        color_pool = [
            QColor(255, 0, 0),
            QColor(255, 0, 0)
            # QColor(0, 255, 0),
            # QColor(0, 0, 255),
            # QColor(255, 255, 0),
            # QColor(255, 0, 255),
        ]
        self.class_colors = {i + 1: color_pool[i] for i in range(num_classes)}
        self.color_names = {1: "Up (Right Arm)",2: "Down (Left Arm)"}
                            #  2: "Green", 3: "Blue", 4: "Yellow", 5: "Magenta"}

        self.trial_order = [c for c in range(1, num_classes + 1) for _ in range(trials_per_class)]
        random.shuffle(self.trial_order)
        self.total_trials = len(self.trial_order)
        self.current_trial = 0
        self.current_class = 0
        self.phase = "idle"
        self.is_running = False

        # Build output folder: data/Subject <N>/<num_classes>_class/
        subject_folder = PROJECT_ROOT / "data" / f"Subject {self.subject_no}"
        self.output_dir = subject_folder / f"{num_classes}_class"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Also ensure the other class folder exists
        other_classes = 3 if num_classes == 2 else 2
        (subject_folder / f"{other_classes}_class").mkdir(parents=True, exist_ok=True)

        self.setWindowTitle(f"EEG Data Collection - Subject {subject_no} - {display_mode.upper()} Mode")
        self.resize(1000, 700)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        
        left_layout.addStretch(1)

        self.info_label = QLabel(f"Subject {subject_no} | {num_classes} Classes | {trials_per_class} Trials\n\nPress 'Start' to begin")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet(
            "font-size: 28px; font-weight: 600; color: white; "
            "padding: 20px; background-color: rgba(0, 0, 0, 0);"
        )
        self.info_label.setWordWrap(True)
        left_layout.addWidget(self.info_label)
        
        left_layout.addSpacing(30)

        self.progress = QProgressBar()
        self.progress.setAlignment(Qt.AlignCenter)
        self.progress.setStyleSheet(
            "QProgressBar { border: 2px solid white; border-radius: 5px; "
            "text-align: center; background-color: #1a1a1a; color: white; }"
            "QProgressBar::chunk { background-color: #4CAF50; }"
        )
        self.progress.setMinimumHeight(30)
        left_layout.addWidget(self.progress)
        
        left_layout.addSpacing(30)

        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet(
            "font-size: 18px; padding: 15px; background-color: #4CAF50; "
            "color: white; border: none; border-radius: 5px;"
        )
        self.start_button.clicked.connect(self.start_experiment)
        left_layout.addWidget(self.start_button)

        left_layout.addStretch(2)

        self.color_display = create_color_display(display_mode, self.class_colors)
        self.color_display.hide()

        main_layout.addLayout(left_layout, 5)
        main_layout.addWidget(self.color_display, 1)
        self.setLayout(main_layout)

        self.set_background_color(QColor(0, 0, 0))

        filename = f"EEG_{display_mode}_trials_{trials_per_class}_classes_{num_classes}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = self.output_dir / filename
        self.log_file = open(filepath, "w", newline="")
        self.csv_writer = csv.writer(self.log_file)
        
        header = ["trial_index", "class_label", "phase", "timestamp"]
        header += [f"Ch{i+1}" for i in range(17)]
        self.csv_writer.writerow(header)
        print(f"CSV will be saved to: {filepath}")

        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)

        if USE_UNICORN:
            self.unicorn_thread = UnicornRecorder(self)
        else:
            self.unicorn_thread = MockEEGRecorder(self)

    def start_experiment(self):
        if self.is_running:
            return
        self.is_running = True
        self.start_button.setEnabled(False)
        self.info_label.setText("Connecting to EEG device...")
        
        if USE_UNICORN:
            try:
                self.unicorn_thread.connect()
                self.unicorn_thread.start()
                self.info_label.setText("Connected! Starting stabilization...")
            except Exception as e:
                print("EEG connection error:", e)
                self.info_label.setText(f"Connection failed: {e}")
                self.is_running = False
                self.start_button.setEnabled(True)
                return
        else:
            self.unicorn_thread.start()
            print("Running GUI with mock EEG recorder.")
            self.info_label.setText("Mock mode - Starting stabilization...")
        
        # Start stabilization phase instead of going directly to trials
        QTimer.singleShot(1000, self.show_stabilization)

    def show_stabilization(self):
        """NEW: Stabilization phase - no data recording"""
        self.phase = "stabilization"
        self.current_class = 0
        self.current_trial = 0
        self.color_display.hide()
        self.phase_duration = self.stabilization_ms
        self.phase_elapsed = 0
        self.progress.setValue(0)
        self.set_background_color(QColor(0, 0, 0))
        
        # Show countdown in seconds
        remaining_seconds = self.stabilization_ms // 1000
        self.info_label.setText(f"Stabilization Phase\n\nPlease relax and minimize movement\n\n{remaining_seconds}s remaining")
        
        # Update countdown every second
        self.stabilization_timer = QTimer()
        self.stabilization_timer.timeout.connect(self.update_stabilization_countdown)
        self.stabilization_timer.start(1000)
        
        self.progress_timer.start(100)
        
        # After stabilization, enable recording and start trials
        QTimer.singleShot(self.stabilization_ms, self.end_stabilization)

    def update_stabilization_countdown(self):
        """Update the countdown display during stabilization"""
        elapsed_seconds = self.phase_elapsed // 1000
        remaining_seconds = (self.stabilization_ms - self.phase_elapsed) // 1000
        if remaining_seconds > 0:
            self.info_label.setText(f"Stabilization Phase\n\nPlease relax and minimize movement\n\n{remaining_seconds}s remaining")

    def end_stabilization(self):
        """End stabilization and begin data recording"""
        self.stabilization_timer.stop()
        print("\n" + "="*50)
        print("STABILIZATION COMPLETE - Beginning data recording")
        print("="*50 + "\n")
        
        # Enable data recording in the thread
        self.unicorn_thread.start_recording()
        
        # Start the actual trials
        self.run_next_trial()

    def run_next_trial(self):
        if self.current_trial >= self.total_trials:
            self.end_experiment()
            return
        self.current_class = self.trial_order[self.current_trial]
        self.current_trial += 1
        self.show_baseline()

    def show_baseline(self):
        self.phase = "baseline"
        self.color_display.hide()
        self.phase_duration = self.baseline_ms
        self.info_label.setText(f"Trial {self.current_trial}/{self.total_trials}\n\nBaseline")
        self.phase_elapsed = 0
        self.progress.setValue(0)
        self.set_background_color(QColor(0, 0, 0))
        self.color_display.reset_colors()
        self.progress_timer.start(100)
        QTimer.singleShot(self.baseline_ms, self.show_instruction)

    def show_instruction(self):
        self.phase = "instruction"
        self.color_display.hide()
        self.phase_duration = self.instruction_display_ms
        color_name = self.color_names[self.current_class]
        self.info_label.setText(f"Think about color:\n\n{color_name}")
        self.phase_elapsed = 0
        self.progress.setValue(0)
        self.set_background_color(QColor(0, 0, 0))
        self.color_display.reset_colors()
        self.progress_timer.start(100)
        QTimer.singleShot(self.instruction_display_ms, self.show_stimulus)

    def show_stimulus(self):
        self.phase = "stimulus"
        self.color_display.show()
        self.phase_duration = self.stim_ms
        self.phase_elapsed = 0
        self.progress.setValue(0)
        self.set_background_color(QColor(0, 0, 0))
        self.info_label.setText(f"Stimulus\n\nClass {self.current_class}")
        self.color_display.highlight_color(self.current_class)
        self.progress_timer.start(100)
        QTimer.singleShot(self.stim_ms, self.run_next_trial)

    def update_progress(self):
        self.phase_elapsed += 100
        val = int((self.phase_elapsed / self.phase_duration) * 100)
        self.progress.setValue(min(val, 100))
        if val >= 100:
            self.progress_timer.stop()

    def set_background_color(self, color: QColor):
        pal = self.palette()
        pal.setColor(QPalette.Window, color)
        self.setAutoFillBackground(True)
        self.setPalette(pal)

    def end_experiment(self):
        self.phase = "done"
        self.progress_timer.stop()
        self.color_display.hide()
        
        if self.unicorn_thread:
            self.unicorn_thread.stop()
            self.unicorn_thread.join(timeout=2)

        self.log_file.close()
        self.set_background_color(QColor(0, 0, 0))
        self.info_label.setText(f"EEG data collection completed!\n\nSaved to:\nSubject {self.subject_no}/{self.num_classes}_class/")
        self.progress.setValue(100)
        print(f"CSV saved to {self.output_dir} and experiment ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Data Collection GUI")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--baseline", type=int, default=1000)
    parser.add_argument("--instruction", type=int, default=2000)
    parser.add_argument("--stimulus", type=int, default=5000)
    parser.add_argument("--stabilization", type=int, default=20000)
    parser.add_argument("--display", type=str, default="bar", choices=["bar", "shapes"])
    args = parser.parse_args()

    app = QApplication(sys.argv)
    gui = EEGTrialGUI(
        num_classes=args.classes,
        trials_per_class=args.trials,
        baseline_ms=args.baseline,
        instruction_display_ms=args.instruction,
        stim_ms=args.stimulus,
        stabilization_ms=args.stabilization,
        display_mode=args.display,
        subject_no=args.subject,
    )
    gui.show()
    sys.exit(app.exec_())