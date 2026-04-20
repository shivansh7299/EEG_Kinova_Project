# # import sys
# # import random
# # import csv
# # from datetime import datetime
# # from PyQt5.QtCore import QTimer, Qt
# # from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
# # from PyQt5.QtGui import QColor, QPalette

# # class EEGTrialGUI(QWidget):
# #     def __init__(self, num_classes=5, trials_per_class=50, baseline_ms=1000, stim_ms=4000, instruction_display_ms = 1000):
# #         super().__init__()

# #         # ---- Config ----
# #         self.num_classes = num_classes           # set to 3 or 5
# #         self.trials_per_class = trials_per_class # e.g., 50 or 80
# #         self.baseline_ms = baseline_ms           # 2 seconds
# #         self.instruction_display_ms = instruction_display_ms
# #         self.stim_ms = stim_ms                   # 4 seconds

# #         # Color pool (use first num_classes)
# #         color_pool = [
# #             QColor(255, 0, 0),     # 1 Red
# #             QColor(0, 255, 0),     # 2 Green
# #             QColor(0, 0, 255),     # 3 Blue
# #             QColor(255, 255, 0),   # 4 Yellow
# #             QColor(255, 0, 255),   # 5 Magenta
# #         ]
# #         assert 1 <= self.num_classes <= len(color_pool), "num_classes must be between 1 and 5"
# #         self.class_colors = {i+1: color_pool[i] for i in range(self.num_classes)}

# #         # Randomized trial order
# #         self.trial_order = []
# #         for c in range(1, self.num_classes + 1):
# #             self.trial_order += [c] * self.trials_per_class
# #         random.shuffle(self.trial_order)

# #         self.total_trials = len(self.trial_order)
# #         self.current_trial = 0
# #         self.current_class = None
# #         self.is_running = False

# #         # ---- UI ----
# #         self.setWindowTitle("EEG Data Collection GUI")
# #         self.resize(900, 700)

# #         layout = QVBoxLayout()
# #         self.info_label = QLabel("Press 'Start' to begin data collection")
# #         self.info_label.setAlignment(Qt.AlignCenter)
# #         self.info_label.setStyleSheet("font-size: 22px; font-weight: 600;")
# #         layout.addWidget(self.info_label)

# #         self.start_button = QPushButton("Start")
# #         self.start_button.setStyleSheet("font-size: 18px; padding: 10px;")
# #         self.start_button.clicked.connect(self.start_experiment)
# #         layout.addWidget(self.start_button)

# #         self.setLayout(layout)
# #         self.set_background_color(QColor(60, 60, 60))  # neutral dark gray

# #         # ---- Logging ----
# #         self.log_file = open(f"eeg_trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "w", newline="")
# #         self.csv_writer = csv.writer(self.log_file)
# #         self.csv_writer.writerow(["trial_index", "class_label", "phase", "timestamp"])

# #         self.show()

# #     def start_experiment(self):
# #         if self.is_running:
# #             return
# #         self.is_running = True
# #         self.start_button.setEnabled(False)
# #         self.info_label.setText("Starting EEG data collection...")
# #         QTimer.singleShot(600, self.run_next_trial)

# #     def run_next_trial(self):
# #         if self.current_trial >= self.total_trials:
# #             self.end_experiment()
# #             return

# #         # Setup for next trial
# #         self.current_class = self.trial_order[self.current_trial]
# #         self.current_trial += 1

# #         # Baseline phase
# #         QTimer.singleShot(self.baseline_ms, self.show_baseline)
# #         QTimer.singleShot(self.instruction_display_ms, self.show_instruction)
            
# #         QTimer.singleShot(self.stim_ms, self.show_stimulus)

# #     def show_baseline(self):
# #         baseline_s = int(self.baseline_ms/1000)
# #         self.set_background_color(QColor(0, 0, 0))  # neutral black
# #         self.info_label.setText(
# #             f"Trial {self.current_trial}/{self.total_trials}\n"
# #             f"Baseline: {baseline_s}s"
# #         )
# #         self.log_event("baseline")
# #         print("Baseline : ", baseline_s)

# #     def show_instruction(self):
# #         instruction_display_s = int(self.instruction_display_ms/1000)
# #         self.set_background_color(QColor(255, 255, 255))  # neutral - something just wanted to put something XD
# #         self.info_label.setText(
# #             f"Trial {self.current_trial}/{self.total_trials}\n"
# #             f"Instrunction: {instruction_display_s}s"
# #         )
# #         self.log_event("Instruction")
# #         print("Instruction : ",instruction_display_s)

# #     def show_stimulus(self):
# #         # Stimulus phase
# #         color = self.class_colors[self.current_class]
# #         self.set_background_color(color)
# #         self.info_label.setText(
# #             f"Trial {self.current_trial}/{self.total_trials}\n"
# #             f"Stimulus (Class {self.current_class}): {self.stim_ms/1000:.1f}s"
# #         )
# #         self.log_event("stimulus")
# #         print("Stimulus")
# #         QTimer.singleShot(self.stim_ms, self.run_next_trial)

# #     def set_background_color(self, color: QColor):
# #         pal = self.palette()
# #         pal.setColor(QPalette.Window, color)
# #         self.setAutoFillBackground(True)
# #         self.setPalette(pal)

# #     def log_event(self, phase: str):
# #         ts = datetime.now().strftime("%H:%M:%S.%f")
# #         # Note: current_trial is 1-based index of the trial being shown
# #         self.csv_writer.writerow([self.current_trial, self.current_class, phase, ts])
# #         self.log_file.flush()

# #     def end_experiment(self):
# #         self.log_file.close()
# #         self.set_background_color(QColor(0, 0, 0))
# #         self.info_label.setText("Data collection completed!")

# # if __name__ == "__main__":
# #     app = QApplication(sys.argv)

# #     # Examples:
# #     # 5 classes × 50 trials each
# #     gui = EEGTrialGUI(num_classes=5, trials_per_class=50, baseline_ms=1000, stim_ms=4000, instruction_display_ms =1000)

# #     # Or for 3 classes × 80 trials each, use:
# #     # gui = EEGTrialGUI(num_classes=3, trials_per_class=80, baseline_ms=2000, stim_ms=4000)

# #     sys.exit(app.exec_())


# # import sys
# # import random
# # import csv
# # from datetime import datetime
# # from PyQt5.QtCore import QTimer, Qt
# # from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QProgressBar
# # from PyQt5.QtGui import QColor, QPalette

# # class EEGTrialGUI(QWidget):
# #     def __init__(self, num_classes=5, trials_per_class=50, baseline_ms=2000, instruction_display_ms=1000, stim_ms=4000):
# #         super().__init__()

# #         self.num_classes = num_classes
# #         self.trials_per_class = trials_per_class
# #         self.baseline_ms = baseline_ms
# #         self.instruction_display_ms = instruction_display_ms
# #         self.stim_ms = stim_ms

# #         # ---- Color Setup ----
# #         color_pool = [
# #             QColor(255, 0, 0),     # Red
# #             QColor(0, 255, 0),     # Green
# #             QColor(0, 0, 255),     # Blue
# #             QColor(255, 255, 0),   # Yellow
# #             QColor(255, 0, 255),   # Magenta
# #         ]
# #         self.class_colors = {i+1: color_pool[i] for i in range(num_classes)}
# #         self.color_names = {
# #             1: "Red",
# #             2: "Green",
# #             3: "Blue",
# #             4: "Yellow",
# #             5: "Pink" #Its not pink its magenta but still. 
# #         }

# #         # ---- Trial Order ----
# #         self.trial_order = []
# #         for c in range(1, num_classes + 1):
# #             self.trial_order += [c] * self.trials_per_class
# #         random.shuffle(self.trial_order)
# #         self.total_trials = len(self.trial_order)
# #         self.current_trial = 0
# #         self.is_running = False

# #         # ---- UI ----
# #         self.setWindowTitle("EEG Data Collection GUI")
# #         self.resize(900, 700)

# #         layout = QVBoxLayout()
# #         self.info_label = QLabel("Press 'Start' to begin data collection")
# #         self.info_label.setAlignment(Qt.AlignCenter)
# #         self.info_label.setStyleSheet("font-size: 22px; font-weight: 600;")
# #         layout.addWidget(self.info_label)

# #         self.progress = QProgressBar()
# #         self.progress.setAlignment(Qt.AlignCenter)
# #         self.progress.setStyleSheet("QProgressBar {font-size: 18px; height: 30px;}")
# #         layout.addWidget(self.progress)

# #         self.start_button = QPushButton("Start")
# #         self.start_button.setStyleSheet("font-size: 18px; padding: 10px;")
# #         self.start_button.clicked.connect(self.start_experiment)
# #         layout.addWidget(self.start_button)

# #         self.setLayout(layout)
# #         self.set_background_color(QColor(60, 60, 60))

# #         self.log_file = open(f"eeg_trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "w", newline="")
# #         self.csv_writer = csv.writer(self.log_file)
# #         self.csv_writer.writerow(["trial_index", "class_label", "phase", "timestamp"])

# #         self.show()

# #         self.progress_timer = QTimer()
# #         self.progress_timer.timeout.connect(self.update_progress)


# #     def start_experiment(self):
# #         if self.is_running:
# #             return
# #         self.is_running = True
# #         self.start_button.setEnabled(False)
# #         self.info_label.setText("Starting EEG data collection...")
# #         QTimer.singleShot(1000, self.run_next_trial)

# #     def run_next_trial(self):
# #         if self.current_trial >= self.total_trials:
# #             self.end_experiment()
# #             return

# #         self.current_class = self.trial_order[self.current_trial]
# #         self.current_trial += 1
# #         self.show_baseline()

# #     def show_baseline(self):
# #         self.phase = "baseline"
# #         self.phase_duration = self.baseline_ms
# #         self.phase_elapsed = 0
# #         self.progress.setValue(0)

# #         self.set_background_color(QColor(0, 0, 0))
# #         self.info_label.setText(
# #             f"Trial {self.current_trial}/{self.total_trials}\nBaseline ({self.baseline_ms/1000:.1f}s)"
# #         )
# #         self.log_event("baseline")

# #         self.progress_timer.start(100)  # update every 100 ms
# #         QTimer.singleShot(self.baseline_ms, self.show_instruction)

# #     def show_instruction(self):
# #         self.phase = "instruction"
# #         self.phase_duration = self.instruction_display_ms
# #         self.phase_elapsed = 0
# #         self.progress.setValue(0)

# #         self.set_background_color(QColor(0, 0, 0))
# #         color_name = self.color_names[self.current_class]
# #         print("Color: ", color_name)
# #         self.info_label.setText(
# #             f"Trial {self.current_trial}/{self.total_trials}\nThink about color : {color_name} ({self.instruction_display_ms/1000:.1f}s)"
# #         )
# #         self.log_event("instruction")
# #         self.progress_timer.start(100) 
# #         QTimer.singleShot(self.instruction_display_ms, self.show_stimulus)

# #     def show_stimulus(self):
# #         self.phase = "stimulus"
# #         self.phase_duration = self.stim_ms
# #         self.phase_elapsed = 0
# #         self.progress.setValue(0)

# #         color = self.class_colors[self.current_class]
# #         print("Here")
# #         self.set_background_color(color)
# #         self.info_label.setText(
# #             f"Trial {self.current_trial}/{self.total_trials}\nStimulus (Class {self.current_class}) {self.stim_ms/1000:.1f}s"
# #         )
# #         self.log_event("stimulus")
# #         self.progress_timer.start(100) 
# #         QTimer.singleShot(self.stim_ms, self.run_next_trial)

# #     def update_progress(self):
# #         if not hasattr(self, "phase_duration"):
# #             return
# #         self.phase_elapsed += 100
# #         progress_percent = min(int((self.phase_elapsed / self.phase_duration) * 100), 100)
# #         self.progress.setValue(progress_percent)

# #         if progress_percent >= 100:
# #             self.progress_timer.stop()

# #     # ----------------- Helpers -----------------
# #     def set_background_color(self, color: QColor):
# #         pal = self.palette()
# #         pal.setColor(QPalette.Window, color)
# #         self.setAutoFillBackground(True)
# #         self.setPalette(pal)

# #     def log_event(self, phase: str):
# #         ts = datetime.now().strftime("%H:%M:%S.%f")
# #         self.csv_writer.writerow([self.current_trial, self.current_class, phase, ts])
# #         self.log_file.flush()

# #     def end_experiment(self):
# #         self.progress_timer.stop()
# #         self.log_file.close()
# #         self.set_background_color(QColor(0, 0, 0))
# #         self.info_label.setText("completed!")
# #         self.progress.setValue(100)

# # if __name__ == "__main__":
# #     app = QApplication(sys.argv)
# #     gui = EEGTrialGUI(num_classes=2, trials_per_class=2, baseline_ms=6000, instruction_display_ms=6000, stim_ms=6000)
# #     sys.exit(app.exec_())

# # import sys, random, csv, threading, time
# # from datetime import datetime
# # from PyQt5.QtCore import QTimer, Qt, QRect
# # from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QProgressBar, QHBoxLayout
# # from PyQt5.QtGui import QColor, QPalette, QPainter
# # import UnicornPy
# # import array
# # import random
# # import ctypes

# # # Toggle EEG hardware integration
# # USE_UNICORN = True


# # # ---------- Unicorn Recorder Thread ----------
# # class UnicornRecorder(threading.Thread):
# #     def __init__(self, gui_ref):
# #         super().__init__(daemon=True)
# #         self.gui = gui_ref
# #         self.running = False
# #         self.device = None
# #         self.fs = 250
# #         self.channels = 8

# #     def connect(self):
# #         devices = UnicornPy.GetAvailableDevices(True)
# #         if not devices:
# #             raise RuntimeError("No Unicorn device found.")
        
# #         print(f"Connecting to device: {devices[0]}")
# #         self.device = UnicornPy.Unicorn(devices[0])

# #         try:
# #             self.device.StartAcquisition(False)
# #             print("Acquisition started successfully.")
# #         except Exception as e:
# #             print("Failed to start acquisition:", e)
# #             raise
        
# #     # def run(self):
# #     #     if UnicornPy is None or self.device is None:
# #     #         return
# #     #     self.running = True
# #     #     block_size = self.fs
# #     #     num_channels = self.device.GetNumberOfAcquiredChannels()
# #     #     buffer = [0] * (block_size * num_channels)
# #     #     writer = self.gui.csv_writer

# #     #     while self.running:
# #     #         self.device.GetData(block_size, buffer, len(buffer))
# #     #         for i in range(block_size):
# #     #             trial = self.gui.current_trial
# #     #             cls = self.gui.current_class
# #     #             phase = getattr(self.gui, "phase", "idle")
# #     #             ts = datetime.now().strftime("%H:%M:%S.%f")
# #     #             row = [trial, cls, phase, ts] + buffer[i*num_channels:(i+1)*num_channels]
# #     #             writer.writerow(row)
# #     #         self.gui.log_file.flush()

# #     #     print("EEG recording stopped.")


# #     def run(self):
# #         if self.device is None:
# #             print("No device connected.")
# #             return

# #         self.running = True
# #         block_size = 50  # samples per read
# #         num_channels = self.device.GetNumberOfAcquiredChannels()
# #         total_floats = block_size * num_channels

# #         # Create a ctypes float array (contiguous memory)
# #         BufferType = ctypes.c_float * total_floats
# #         buffer = BufferType()

# #         print(f"Unicorn recording started: {num_channels} channels, block size {block_size}")

# #         while self.running:
# #             try:
# #                 # GetData expects: (frame_count, buffer, buffer_size_in_bytes)
# #                 self.device.GetData(block_size, buffer, ctypes.sizeof(buffer))
                
# #                 # Convert to Python list
# #                 data = list(buffer)
                
# #                 # Write each sample as a row
# #                 for i in range(block_size):
# #                     trial = self.gui.current_trial
# #                     cls = self.gui.current_class
# #                     phase = getattr(self.gui, "phase", "idle")
# #                     ts = datetime.now().strftime("%H:%M:%S.%f")
                    
# #                     # Extract channel data for this sample
# #                     sample_data = data[i * num_channels : (i + 1) * num_channels]
# #                     row = [trial, cls, phase, ts] + sample_data
                    
# #                     self.gui.csv_writer.writerow(row)
                
# #                 # Flush after each block
# #                 self.gui.log_file.flush()
                
# #             except Exception as e:
# #                 print(f"Error in GetData: {e}")
# #                 break

# #         print("EEG recording stopped.")



# #     def stop(self):
# #         self.running = False
# #         if self.device:
# #             self.device.StopAcquisition()
# #             self.device.Disconnect()
# #             print("Disconnected from Unicorn.")

# # class MockEEGRecorder(threading.Thread):
# #     def __init__(self, gui_ref):
# #         super().__init__(daemon=True)
# #         self.gui = gui_ref
# #         self.running = False

# #     def run(self):
# #         self.running = True
# #         while self.running:
# #             fake_data = [random.uniform(-100, 100) for _ in range(8)]
# #             ts = datetime.now().strftime("%H:%M:%S.%f")
# #             self.gui.csv_writer.writerow([self.gui.current_trial, self.gui.current_class, self.gui.phase, ts] + fake_data)
# #             self.gui.log_file.flush()
# #             time.sleep(1 / 250)  # mimic 250 Hz



# # # ---------- Color Bar Widget ----------
# # class ColorBar(QWidget):
# #     def __init__(self, class_colors):
# #         super().__init__()
# #         self.class_colors = class_colors
# #         self.highlight_index = None  # which color to highlight
# #         self.setMinimumWidth(80)

# #     def paintEvent(self, event):
# #         painter = QPainter(self)
# #         bar_width = self.width()
# #         bar_height = self.height() // len(self.class_colors)

# #         for i, (cls, color) in enumerate(self.class_colors.items()):
# #             rect = QRect(0, i * bar_height, bar_width, bar_height)
# #             # If highlighted, draw bright color; else greyed out
# #             if self.highlight_index is None:
# #                 painter.fillRect(rect, color)
# #             elif cls == self.highlight_index:
# #                 painter.fillRect(rect, color)
# #             else:
# #                 painter.fillRect(rect, QColor(30, 30, 30))  # greyed out
# #         painter.end()

# #     def highlight_color(self, class_index):
# #         """Highlight one color (during stimulus)."""
# #         self.highlight_index = class_index
# #         self.update()

# #     def reset_colors(self):
# #         """Show all class colors (during instruction)."""
# #         self.highlight_index = None
# #         self.update()


# # # ---------- Main EEG GUI ----------
# # class EEGTrialGUI(QWidget):
# #     def __init__(self, num_classes=5, trials_per_class=3, baseline_ms=3000, instruction_display_ms=3000, stim_ms=3000):
# #         super().__init__()

# #         self.num_classes = num_classes
# #         self.trials_per_class = trials_per_class
# #         self.baseline_ms = baseline_ms
# #         self.instruction_display_ms = instruction_display_ms
# #         self.stim_ms = stim_ms

# #         # ---- Class Colors ----
# #         color_pool = [
# #             QColor(255, 0, 0),     # Red
# #             QColor(0, 255, 0),     # Green
# #             QColor(0, 0, 255),     # Blue
# #             QColor(255, 255, 0),   # Yellow
# #             QColor(255, 0, 255),   # Magenta
# #         ]
# #         self.class_colors = {i + 1: color_pool[i] for i in range(num_classes)}
# #         self.color_names = {1: "Red", 2: "Green", 3: "Blue", 4: "Yellow", 5: "Magenta"}

# #         # ---- Trials ----
# #         self.trial_order = [c for c in range(1, num_classes + 1) for _ in range(trials_per_class)]
# #         random.shuffle(self.trial_order)
# #         self.total_trials = len(self.trial_order)
# #         self.current_trial = 0
# #         self.current_class = 0
# #         self.phase = "idle"
# #         self.is_running = False

# #         # ---- UI ----
# #         self.setWindowTitle("EEG Data Collection GUI")
# #         self.resize(1000, 700)

# #         main_layout = QHBoxLayout()
# #         left_layout = QVBoxLayout()

# #         self.info_label = QLabel("Press 'Start' to begin data collection")
# #         self.info_label.setAlignment(Qt.AlignCenter)
# #         self.info_label.setStyleSheet("font-size: 22px; font-weight: 600;")
# #         left_layout.addWidget(self.info_label)

# #         self.progress = QProgressBar()
# #         self.progress.setAlignment(Qt.AlignCenter)
# #         left_layout.addWidget(self.progress)

# #         self.start_button = QPushButton("Start")
# #         self.start_button.setStyleSheet("font-size: 18px; padding: 10px;")
# #         self.start_button.clicked.connect(self.start_experiment)
# #         left_layout.addWidget(self.start_button)

# #         left_layout.addStretch()

# #         # ---- Color Bar on the Right ----
# #         self.color_bar = ColorBar(self.class_colors)

# #         main_layout.addLayout(left_layout, 4)
# #         main_layout.addWidget(self.color_bar, 1)
# #         self.setLayout(main_layout)

# #         self.set_background_color(QColor(0, 0, 0))

# #         # ---- CSV ----
# #         filename = f"EEG_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
# #         self.log_file = open(filename, "w", newline="")
# #         self.csv_writer = csv.writer(self.log_file)
# #         self.csv_writer.writerow(["trial_index", "class_label", "phase", "timestamp",
# #                                   "Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7", "Ch8"])

# #         self.progress_timer = QTimer()
# #         self.progress_timer.timeout.connect(self.update_progress)

# #         # ---- EEG Thread ----
# #         if USE_UNICORN:
# #             self.unicorn_thread = UnicornRecorder(self)
# #         else:
# #             # self.unicorn_thread = None
# #             self.unicorn_thread = MockEEGRecorder(self)

# #     # ---------- Experiment Flow ----------
# #     def start_experiment(self):
# #         if self.is_running:
# #             return
# #         self.is_running = True
# #         self.start_button.setEnabled(False)
# #         self.info_label.setText("Connecting to EEG device...")
# #         if USE_UNICORN:
# #             try:
# #                 self.unicorn_thread.connect()
# #                 self.unicorn_thread.start()
# #             except Exception as e:
# #                 print("EEG connection error:", e)
# #         else:
# #             print("Running GUI without Unicorn device.")
# #         # if self.is_running:
# #         #     return
# #         # self.is_running = True
# #         # self.start_button.setEnabled(False)
# #         # self.info_label.setText("Starting experiment...")

# #         # if self.unicorn_thread:
# #         #     self.unicorn_thread.start()   # ✅ this line is key!
# #         #     print("⚙️ Running GUI with mock EEG recorder.")
# #         QTimer.singleShot(1000, self.run_next_trial)

# #     def run_next_trial(self):
# #         if self.current_trial >= self.total_trials:
# #             self.end_experiment()
# #             return
# #         self.current_class = self.trial_order[self.current_trial]
# #         self.current_trial += 1
# #         self.show_baseline()

# #     def show_baseline(self):
# #         self.phase = "baseline"
# #         self.color_bar.hide()
# #         self.phase_duration = self.baseline_ms
# #         self.phase_elapsed = 0
# #         self.progress.setValue(0)
# #         self.set_background_color(QColor(0, 0, 0))
# #         self.color_bar.reset_colors()
# #         self.info_label.setText(f"Trial {self.current_trial}/{self.total_trials}\nBaseline")
# #         self.progress_timer.start(100)
# #         QTimer.singleShot(self.baseline_ms, self.show_instruction)

# #     def show_instruction(self):
# #         self.phase = "instruction"
# #         self.color_bar.hide()
# #         self.phase_duration = self.instruction_display_ms
# #         self.phase_elapsed = 0
# #         self.progress.setValue(0)
# #         self.set_background_color(QColor(0, 0, 0))
# #         color_name = self.color_names[self.current_class]
# #         self.info_label.setText(f"Think about color: {color_name}")
# #         self.color_bar.reset_colors()
# #         self.progress_timer.start(100)
# #         QTimer.singleShot(self.instruction_display_ms, self.show_stimulus)

# #     def show_stimulus(self):
# #         self.phase = "stimulus"
# #         self.color_bar.show()
# #         self.phase_duration = self.stim_ms
# #         self.phase_elapsed = 0
# #         self.progress.setValue(0)
# #         self.set_background_color(QColor(0, 0, 0))
# #         self.info_label.setText(f"Stimulus (Class {self.current_class})")
# #         self.color_bar.highlight_color(self.current_class)
# #         self.progress_timer.start(100)
# #         QTimer.singleShot(self.stim_ms, self.run_next_trial)

# #     def update_progress(self):
# #         self.phase_elapsed += 100
# #         val = int((self.phase_elapsed / self.phase_duration) * 100)
# #         self.progress.setValue(min(val, 100))
# #         if val >= 100:
# #             self.progress_timer.stop()

# #     def set_background_color(self, color: QColor):
# #         pal = self.palette()
# #         pal.setColor(QPalette.Window, color)
# #         self.setAutoFillBackground(True)
# #         self.setPalette(pal)

# #     def end_experiment(self):
# #         self.phase = "done"
# #         self.progress_timer.stop()
# #         if USE_UNICORN and self.unicorn_thread:
# #             self.unicorn_thread.stop()

# #         self.log_file.close()
# #         self.set_background_color(QColor(0, 0, 0))
# #         self.info_label.setText("EEG data collection completed!")
# #         self.progress.setValue(100)
# #         print("Unified CSV saved and experiment ended.")


# # if __name__ == "__main__":
# #     app = QApplication(sys.argv)
# #     gui = EEGTrialGUI(num_classes=2, trials_per_class=2, baseline_ms=3000, instruction_display_ms=3000, stim_ms=3000)
# #     gui.show()
# #     sys.exit(app.exec_())


# # import sys, random, csv, threading, time
# # from datetime import datetime
# # from PyQt5.QtCore import QTimer, Qt, QRect
# # from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QProgressBar, QHBoxLayout
# # from PyQt5.QtGui import QColor, QPalette, QPainter
# # import UnicornPy
# # import numpy as np

# # # Toggle EEG hardware integration
# # USE_UNICORN = True


# # # ---------- Unicorn Recorder Thread ----------
# # class UnicornRecorder(threading.Thread):
# #     def __init__(self, gui_ref):
# #         super().__init__(daemon=True)
# #         self.gui = gui_ref
# #         self.running = False
# #         self.device = None
# #         self.fs = 250
# #         self.num_channels = None

# #     def connect(self):
# #         devices = UnicornPy.GetAvailableDevices(True)
# #         if not devices:
# #             raise RuntimeError("No Unicorn device found.")
        
# #         print(f"Connecting to device: {devices[0]}")
# #         self.device = UnicornPy.Unicorn(devices[0])
        
# #         self.num_channels = self.device.GetNumberOfAcquiredChannels()
# #         print(f"Device has {self.num_channels} channels")
# #         print(f"Sample rate: {self.fs} Hz")
        
# #         try:
# #             self.device.StartAcquisition(False)
# #             print("Acquisition started successfully.")
# #         except Exception as e:
# #             print("Failed to start acquisition:", e)
# #             raise
        
# #     def run(self):
# #         if self.device is None:
# #             print("No device connected.")
# #             return

# #         self.running = True
        
# #         # KEY INSIGHT FROM SENIOR'S CODE:
# #         # Read small chunks frequently (like 60Hz updates) instead of big blocks
# #         # This prevents buffer overflow and is more stable
        
# #         samples_per_read = 4  # Read 4 samples at ~60Hz (250Hz / 4 ≈ 60Hz)
# #         buffer_size = samples_per_read * self.num_channels * 4  # 4 bytes per float32
        
# #         print(f"Recording: {self.num_channels} channels, {samples_per_read} samples per read")
# #         print(f"Buffer size: {buffer_size} bytes")
# #         print("Starting data acquisition loop...")

# #         total_samples = 0
# #         last_flush_time = time.time()
        
# #         time.sleep(0.5)
        
# #         try:
# #             while self.running:
# #                 # Create a NEW bytearray for each read (important!)
# #                 receive_buffer = bytearray(buffer_size)
                
# #                 # GetData: (number_of_samples, buffer, buffer_size_in_bytes)
# #                 self.device.GetData(samples_per_read, receive_buffer, buffer_size)
                
# #                 # Convert bytearray to numpy array
# #                 data = np.frombuffer(receive_buffer, dtype=np.float32)
                
# #                 # Reshape to (samples, channels)
# #                 data = data.reshape((samples_per_read, self.num_channels))
                
# #                 # Process each sample in this chunk
# #                 for sample in data:
# #                     total_samples += 1
                    
# #                     # Get trial info
# #                     trial = self.gui.current_trial
# #                     cls = self.gui.current_class
# #                     phase = getattr(self.gui, "phase", "idle")
# #                     ts = datetime.now().strftime("%H:%M:%S.%f")
                    
# #                     # Write one row per sample (250 rows per second)
# #                     row = [trial, cls, phase, ts] + sample.tolist()
# #                     self.gui.csv_writer.writerow(row)
                
# #                 # Flush to disk every second
# #                 current_time = time.time()
# #                 if current_time - last_flush_time >= 1.0:
# #                     self.gui.log_file.flush()
# #                     last_flush_time = current_time
# #                     print(f"✓ {total_samples} samples recorded ({total_samples/250:.1f}s)")
                
# #         except KeyboardInterrupt:
# #             print("Recording interrupted by user")
# #         except Exception as e:
# #             print(f"❌ Fatal error after {total_samples} samples: {e}")
# #             import traceback
# #             traceback.print_exc()

# #         print(f"EEG recording stopped. Total: {total_samples} samples ({total_samples/250:.1f}s)")

# #     def stop(self):
# #         print("Stopping recorder thread...")
# #         self.running = False
# #         time.sleep(0.5)
# #         if self.device:
# #             try:
# #                 self.device.StopAcquisition()
# #                 print("Acquisition stopped.")
# #             except Exception as ex:
# #                 print(f"Stop error: {ex}")
# #             try:
# #                 del self.device
# #                 self.device = None
# #                 print("Device disconnected.")
# #             except Exception as ex:
# #                 print(f"Disconnect error: {ex}")


# # class MockEEGRecorder(threading.Thread):
# #     def __init__(self, gui_ref):
# #         super().__init__(daemon=True)
# #         self.gui = gui_ref
# #         self.running = False
# #         self.num_channels = 17  # Mock the same channel count

# #     def run(self):
# #         self.running = True
# #         print("Running mock EEG recorder (17 channels)")
# #         while self.running:
# #             # Generate fake data for 17 channels
# #             fake_data = [random.uniform(-100, 100) for _ in range(self.num_channels)]
# #             ts = datetime.now().strftime("%H:%M:%S.%f")
# #             trial = self.gui.current_trial
# #             cls = self.gui.current_class
# #             phase = getattr(self.gui, "phase", "idle")
# #             self.gui.csv_writer.writerow([trial, cls, phase, ts] + fake_data)
# #             self.gui.log_file.flush()
# #             time.sleep(1 / 250)  # mimic 250 Hz

# #     def stop(self):
# #         self.running = False


# # # ---------- Color Bar Widget ----------
# # class ColorBar(QWidget):
# #     def __init__(self, class_colors):
# #         super().__init__()
# #         self.class_colors = class_colors
# #         self.highlight_index = None  # which color to highlight
# #         self.setMinimumWidth(80)

# #     def paintEvent(self, event):
# #         painter = QPainter(self)
# #         bar_width = self.width()
# #         bar_height = self.height() // len(self.class_colors)

# #         for i, (cls, color) in enumerate(self.class_colors.items()):
# #             rect = QRect(0, i * bar_height, bar_width, bar_height)
# #             # If highlighted, draw bright color; else greyed out
# #             if self.highlight_index is None:
# #                 painter.fillRect(rect, color)
# #             elif cls == self.highlight_index:
# #                 painter.fillRect(rect, color)
# #             else:
# #                 painter.fillRect(rect, QColor(30, 30, 30))  # greyed out
# #         painter.end()

# #     def highlight_color(self, class_index):
# #         """Highlight one color (during stimulus)."""
# #         self.highlight_index = class_index
# #         self.update()

# #     def reset_colors(self):
# #         """Show all class colors (during instruction)."""
# #         self.highlight_index = None
# #         self.update()


# # # ---------- Main EEG GUI ----------
# # class EEGTrialGUI(QWidget):
# #     def __init__(self, num_classes=5, trials_per_class=3, baseline_ms=3000, instruction_display_ms=3000, stim_ms=3000):
# #         super().__init__()

# #         self.num_classes = num_classes
# #         self.trials_per_class = trials_per_class
# #         self.baseline_ms = baseline_ms
# #         self.instruction_display_ms = instruction_display_ms
# #         self.stim_ms = stim_ms

# #         # ---- Class Colors ----
# #         color_pool = [
# #             QColor(255, 0, 0),     # Red
# #             QColor(0, 255, 0),     # Green
# #             QColor(0, 0, 255),     # Blue
# #             QColor(255, 255, 0),   # Yellow
# #             QColor(255, 0, 255),   # Magenta
# #         ]
# #         self.class_colors = {i + 1: color_pool[i] for i in range(num_classes)}
# #         self.color_names = {1: "Red", 2: "Green", 3: "Blue", 4: "Yellow", 5: "Magenta"}

# #         # ---- Trials ----
# #         self.trial_order = [c for c in range(1, num_classes + 1) for _ in range(trials_per_class)]
# #         random.shuffle(self.trial_order)
# #         self.total_trials = len(self.trial_order)
# #         self.current_trial = 0
# #         self.current_class = 0
# #         self.phase = "idle"
# #         self.is_running = False

# #         # ---- UI ----
# #         self.setWindowTitle("EEG Data Collection GUI")
# #         self.resize(1000, 700)

# #         main_layout = QHBoxLayout()
# #         left_layout = QVBoxLayout()

# #         self.info_label = QLabel("Press 'Start' to begin data collection")
# #         self.info_label.setAlignment(Qt.AlignCenter)
# #         self.info_label.setStyleSheet("font-size: 22px; font-weight: 600;")
# #         left_layout.addWidget(self.info_label)

# #         self.progress = QProgressBar()
# #         self.progress.setAlignment(Qt.AlignCenter)
# #         left_layout.addWidget(self.progress)

# #         self.start_button = QPushButton("Start")
# #         self.start_button.setStyleSheet("font-size: 18px; padding: 10px;")
# #         self.start_button.clicked.connect(self.start_experiment)
# #         left_layout.addWidget(self.start_button)

# #         left_layout.addStretch()

# #         # ---- Color Bar on the Right ----
# #         self.color_bar = ColorBar(self.class_colors)

# #         main_layout.addLayout(left_layout, 4)
# #         main_layout.addWidget(self.color_bar, 1)
# #         self.setLayout(main_layout)

# #         self.set_background_color(QColor(0, 0, 0))

# #         # ---- CSV ----
# #         filename = f"EEG_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
# #         self.log_file = open(filename, "w", newline="")
# #         self.csv_writer = csv.writer(self.log_file)
        
# #         # Unicorn Hybrid Black has 17 channels:
# #         # 8 EEG + 3 Accelerometer + 3 Gyroscope + Battery + Counter + Validation
# #         header = ["trial_index", "class_label", "phase", "timestamp"]
# #         header += [f"Ch{i+1}" for i in range(17)]  # 17 channels
# #         self.csv_writer.writerow(header)

# #         self.progress_timer = QTimer()
# #         self.progress_timer.timeout.connect(self.update_progress)

# #         # ---- EEG Thread ----
# #         if USE_UNICORN:
# #             self.unicorn_thread = UnicornRecorder(self)
# #         else:
# #             self.unicorn_thread = MockEEGRecorder(self)

# #     # ---------- Experiment Flow ----------
# #     def start_experiment(self):
# #         if self.is_running:
# #             return
# #         self.is_running = True
# #         self.start_button.setEnabled(False)
# #         self.info_label.setText("Connecting to EEG device...")
        
# #         if USE_UNICORN:
# #             try:
# #                 self.unicorn_thread.connect()
# #                 self.unicorn_thread.start()
# #                 self.info_label.setText("Connected! Starting experiment...")
# #             except Exception as e:
# #                 print("EEG connection error:", e)
# #                 self.info_label.setText(f"Connection failed: {e}")
# #                 self.is_running = False
# #                 self.start_button.setEnabled(True)
# #                 return
# #         else:
# #             self.unicorn_thread.start()
# #             print("Running GUI with mock EEG recorder.")
# #             self.info_label.setText("Mock mode - Starting experiment...")
        
# #         QTimer.singleShot(1000, self.run_next_trial)

# #     def run_next_trial(self):
# #         if self.current_trial >= self.total_trials:
# #             self.end_experiment()
# #             return
# #         self.current_class = self.trial_order[self.current_trial]
# #         self.current_trial += 1
# #         self.show_baseline()

# #     def show_baseline(self):
# #         self.phase = "baseline"
# #         self.color_bar.hide()
# #         self.phase_duration = self.baseline_ms
# #         self.info_label.setText(f"Trial {self.current_trial}/{self.total_trials}\nBaseline")
# #         self.phase_elapsed = 0
# #         self.progress.setValue(0)
# #         self.set_background_color(QColor(0, 0, 0))
# #         self.color_bar.reset_colors()
# #         self.progress_timer.start(100)
# #         QTimer.singleShot(self.baseline_ms, self.show_instruction)

# #     def show_instruction(self):
# #         self.phase = "instruction"
# #         self.color_bar.hide()
# #         self.phase_duration = self.instruction_display_ms
# #         color_name = self.color_names[self.current_class]
# #         self.info_label.setText(f"Think about color: {color_name}")
# #         self.phase_elapsed = 0
# #         self.progress.setValue(0)
# #         self.set_background_color(QColor(0, 0, 0))
# #         self.color_bar.reset_colors()
# #         self.progress_timer.start(100)
# #         QTimer.singleShot(self.instruction_display_ms, self.show_stimulus)

# #     def show_stimulus(self):
# #         self.phase = "stimulus"
# #         self.color_bar.show()
# #         self.phase_duration = self.stim_ms
# #         self.phase_elapsed = 0
# #         self.progress.setValue(0)
# #         self.set_background_color(QColor(0, 0, 0))
# #         self.info_label.setText(f"Stimulus (Class {self.current_class})")
# #         self.color_bar.highlight_color(self.current_class)
# #         self.progress_timer.start(100)
# #         QTimer.singleShot(self.stim_ms, self.run_next_trial)

# #     def update_progress(self):
# #         self.phase_elapsed += 100
# #         val = int((self.phase_elapsed / self.phase_duration) * 100)
# #         self.progress.setValue(min(val, 100))
# #         if val >= 100:
# #             self.progress_timer.stop()

# #     def set_background_color(self, color: QColor):
# #         pal = self.palette()
# #         pal.setColor(QPalette.Window, color)
# #         self.setAutoFillBackground(True)
# #         self.setPalette(pal)

# #     def end_experiment(self):
# #         self.phase = "done"
# #         self.progress_timer.stop()
        
# #         if self.unicorn_thread:
# #             self.unicorn_thread.stop()
# #             self.unicorn_thread.join(timeout=2)

# #         self.log_file.close()
# #         self.set_background_color(QColor(0, 0, 0))
# #         self.info_label.setText("EEG data collection completed!")
# #         self.progress.setValue(100)
# #         print("CSV saved and experiment ended.")


# # if __name__ == "__main__":
# #     app = QApplication(sys.argv)
# #     gui = EEGTrialGUI(num_classes=2, trials_per_class=2, baseline_ms=1000, instruction_display_ms=1000, stim_ms=5000)
# #     gui.show()
# #     sys.exit(app.exec_())

# import sys, random, csv, threading, time
# from datetime import datetime
# from PyQt5.QtCore import QTimer, Qt, QRect
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QProgressBar, QHBoxLayout
# from PyQt5.QtGui import QColor, QPalette, QPainter
# import UnicornPy
# import numpy as np

# # Toggle EEG hardware integration
# USE_UNICORN = True


# # ---------- Unicorn Recorder Thread ----------
# class UnicornRecorder(threading.Thread):
#     def __init__(self, gui_ref):
#         super().__init__(daemon=True)
#         self.gui = gui_ref
#         self.running = False
#         self.device = None
#         self.fs = 250
#         self.num_channels = None

#     def connect(self):
#         devices = UnicornPy.GetAvailableDevices(True)
#         if not devices:
#             raise RuntimeError("No Unicorn device found.")
        
#         print(f"Connecting to device: {devices[0]}")
#         self.device = UnicornPy.Unicorn(devices[0])
        
#         self.num_channels = self.device.GetNumberOfAcquiredChannels()
#         print(f"Device has {self.num_channels} channels")
#         print(f"Sample rate: {self.fs} Hz")
        
#         try:
#             self.device.StartAcquisition(False)
#             print("Acquisition started successfully.")
#         except Exception as e:
#             print("Failed to start acquisition:", e)
#             raise
        
#     def run(self):
#         if self.device is None:
#             print("No device connected.")
#             return

#         self.running = True
        
#         samples_per_read = 4
#         buffer_size = samples_per_read * self.num_channels * 4
        
#         print(f"Recording: {self.num_channels} channels, {samples_per_read} samples per read")
#         print(f"Buffer size: {buffer_size} bytes")
#         print("Starting data acquisition loop...")

#         total_samples = 0
#         last_flush_time = time.time()
        
#         time.sleep(0.5)
        
#         try:
#             while self.running:
#                 receive_buffer = bytearray(buffer_size)
#                 self.device.GetData(samples_per_read, receive_buffer, buffer_size)
#                 data = np.frombuffer(receive_buffer, dtype=np.float32)
#                 data = data.reshape((samples_per_read, self.num_channels))
                
#                 for sample in data:
#                     total_samples += 1
#                     trial = self.gui.current_trial
#                     cls = self.gui.current_class
#                     phase = getattr(self.gui, "phase", "idle")
#                     ts = datetime.now().strftime("%H:%M:%S.%f")
#                     row = [trial, cls, phase, ts] + sample.tolist()
#                     self.gui.csv_writer.writerow(row)
                
#                 current_time = time.time()
#                 if current_time - last_flush_time >= 1.0:
#                     self.gui.log_file.flush()
#                     last_flush_time = current_time
#                     print(f"✓ {total_samples} samples recorded ({total_samples/250:.1f}s)")
                
#         except KeyboardInterrupt:
#             print("Recording interrupted by user")
#         except Exception as e:
#             print(f"❌ Fatal error after {total_samples} samples: {e}")
#             import traceback
#             traceback.print_exc()

#         print(f"EEG recording stopped. Total: {total_samples} samples ({total_samples/250:.1f}s)")

#     def stop(self):
#         print("Stopping recorder thread...")
#         self.running = False
#         time.sleep(0.5)
#         if self.device:
#             try:
#                 self.device.StopAcquisition()
#                 print("Acquisition stopped.")
#             except Exception as ex:
#                 print(f"Stop error: {ex}")
#             try:
#                 del self.device
#                 self.device = None
#                 print("Device disconnected.")
#             except Exception as ex:
#                 print(f"Disconnect error: {ex}")


# class MockEEGRecorder(threading.Thread):
#     def __init__(self, gui_ref):
#         super().__init__(daemon=True)
#         self.gui = gui_ref
#         self.running = False
#         self.num_channels = 17

#     def run(self):
#         self.running = True
#         print("Running mock EEG recorder (17 channels)")
#         while self.running:
#             fake_data = [random.uniform(-100, 100) for _ in range(self.num_channels)]
#             ts = datetime.now().strftime("%H:%M:%S.%f")
#             trial = self.gui.current_trial
#             cls = self.gui.current_class
#             phase = getattr(self.gui, "phase", "idle")
#             self.gui.csv_writer.writerow([trial, cls, phase, ts] + fake_data)
#             self.gui.log_file.flush()
#             time.sleep(1 / 250)

#     def stop(self):
#         self.running = False


# # ---------- Color Bar Widget ----------
# class ColorBar(QWidget):
#     def __init__(self, class_colors):
#         super().__init__()
#         self.class_colors = class_colors
#         self.highlight_index = None
#         self.setMinimumWidth(80)
#         self.setMaximumWidth(120)

#     def paintEvent(self, event):
#         painter = QPainter(self)
#         bar_width = self.width()
#         bar_height = self.height() // len(self.class_colors)

#         for i, (cls, color) in enumerate(self.class_colors.items()):
#             rect = QRect(0, i * bar_height, bar_width, bar_height)
#             if self.highlight_index is None:
#                 painter.fillRect(rect, color)
#             elif cls == self.highlight_index:
#                 painter.fillRect(rect, color)
#             else:
#                 painter.fillRect(rect, QColor(30, 30, 30))
#         painter.end()

#     def highlight_color(self, class_index):
#         self.highlight_index = class_index
#         self.update()

#     def reset_colors(self):
#         self.highlight_index = None
#         self.update()


# # ---------- Main EEG GUI ----------
# class EEGTrialGUI(QWidget):
#     def __init__(self, num_classes=5, trials_per_class=3, baseline_ms=3000, instruction_display_ms=3000, stim_ms=3000):
#         super().__init__()

#         self.num_classes = num_classes
#         self.trials_per_class = trials_per_class
#         self.baseline_ms = baseline_ms
#         self.instruction_display_ms = instruction_display_ms
#         self.stim_ms = stim_ms

#         # ---- Class Colors ----
#         color_pool = [
#             QColor(255, 0, 0),     # Red
#             QColor(0, 255, 0),     # Green
#             QColor(0, 0, 255),     # Blue
#             QColor(255, 255, 0),   # Yellow
#             QColor(255, 0, 255),   # Magenta
#         ]
#         self.class_colors = {i + 1: color_pool[i] for i in range(num_classes)}
#         self.color_names = {1: "Red", 2: "Green", 3: "Blue", 4: "Yellow", 5: "Magenta"}

#         # ---- Trials ----
#         self.trial_order = [c for c in range(1, num_classes + 1) for _ in range(trials_per_class)]
#         random.shuffle(self.trial_order)
#         self.total_trials = len(self.trial_order)
#         self.current_trial = 0
#         self.current_class = 0
#         self.phase = "idle"
#         self.is_running = False

#         # ---- UI ----
#         self.setWindowTitle("EEG Data Collection GUI")
#         self.resize(1000, 700)

#         # FIXED: Better layout structure
#         main_layout = QHBoxLayout()
#         left_layout = QVBoxLayout()
        
#         # Add spacing at top
#         left_layout.addStretch(1)

#         # FIXED: Better text styling with white color on black background
#         self.info_label = QLabel("Press 'Start' to begin data collection")
#         self.info_label.setAlignment(Qt.AlignCenter)
#         self.info_label.setStyleSheet(
#             "font-size: 28px; font-weight: 600; color: white; "
#             "padding: 20px; background-color: rgba(0, 0, 0, 0);"
#         )
#         self.info_label.setWordWrap(True)
#         left_layout.addWidget(self.info_label)
        
#         left_layout.addSpacing(30)

#         self.progress = QProgressBar()
#         self.progress.setAlignment(Qt.AlignCenter)
#         self.progress.setStyleSheet(
#             "QProgressBar { border: 2px solid white; border-radius: 5px; "
#             "text-align: center; background-color: #1a1a1a; color: white; }"
#             "QProgressBar::chunk { background-color: #4CAF50; }"
#         )
#         self.progress.setMinimumHeight(30)
#         left_layout.addWidget(self.progress)
        
#         left_layout.addSpacing(30)

#         self.start_button = QPushButton("Start")
#         self.start_button.setStyleSheet(
#             "font-size: 18px; padding: 15px; background-color: #4CAF50; "
#             "color: white; border: none; border-radius: 5px;"
#         )
#         self.start_button.clicked.connect(self.start_experiment)
#         left_layout.addWidget(self.start_button)

#         left_layout.addStretch(2)

#         # ---- Color Bar on the Right ----
#         # FIXED: Initially hide the color bar completely
#         self.color_bar = ColorBar(self.class_colors)
#         self.color_bar.hide()  # Start hidden

#         main_layout.addLayout(left_layout, 5)
#         main_layout.addWidget(self.color_bar, 1)
#         self.setLayout(main_layout)

#         self.set_background_color(QColor(0, 0, 0))

#         # ---- CSV ----
#         filename = f"EEG_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
#         self.log_file = open(filename, "w", newline="")
#         self.csv_writer = csv.writer(self.log_file)
        
#         header = ["trial_index", "class_label", "phase", "timestamp"]
#         header += [f"Ch{i+1}" for i in range(17)]
#         self.csv_writer.writerow(header)

#         self.progress_timer = QTimer()
#         self.progress_timer.timeout.connect(self.update_progress)

#         # ---- EEG Thread ----
#         if USE_UNICORN:
#             self.unicorn_thread = UnicornRecorder(self)
#         else:
#             self.unicorn_thread = MockEEGRecorder(self)

#     # ---------- Experiment Flow ----------
#     def start_experiment(self):
#         if self.is_running:
#             return
#         self.is_running = True
#         self.start_button.setEnabled(False)
#         self.info_label.setText("Connecting to EEG device...")
        
#         if USE_UNICORN:
#             try:
#                 self.unicorn_thread.connect()
#                 self.unicorn_thread.start()
#                 self.info_label.setText("Connected! Starting experiment...")
#             except Exception as e:
#                 print("EEG connection error:", e)
#                 self.info_label.setText(f"Connection failed: {e}")
#                 self.is_running = False
#                 self.start_button.setEnabled(True)
#                 return
#         else:
#             self.unicorn_thread.start()
#             print("Running GUI with mock EEG recorder.")
#             self.info_label.setText("Mock mode - Starting experiment...")
        
#         QTimer.singleShot(1000, self.run_next_trial)

#     def run_next_trial(self):
#         if self.current_trial >= self.total_trials:
#             self.end_experiment()
#             return
#         self.current_class = self.trial_order[self.current_trial]
#         self.current_trial += 1
#         self.show_baseline()

#     def show_baseline(self):
#         self.phase = "baseline"
#         # FIXED: Explicitly hide color bar during baseline
#         self.color_bar.hide()
#         self.phase_duration = self.baseline_ms
#         self.info_label.setText(f"Trial {self.current_trial}/{self.total_trials}\n\nBaseline")
#         self.phase_elapsed = 0
#         self.progress.setValue(0)
#         self.set_background_color(QColor(0, 0, 0))
#         self.color_bar.reset_colors()
#         self.progress_timer.start(100)
#         QTimer.singleShot(self.baseline_ms, self.show_instruction)

#     def show_instruction(self):
#         self.phase = "instruction"
#         # FIXED: Keep color bar hidden during instruction
#         self.color_bar.hide()
#         self.phase_duration = self.instruction_display_ms
#         color_name = self.color_names[self.current_class]
#         self.info_label.setText(f"Think about color:\n\n{color_name}")
#         self.phase_elapsed = 0
#         self.progress.setValue(0)
#         self.set_background_color(QColor(0, 0, 0))
#         self.color_bar.reset_colors()
#         self.progress_timer.start(100)
#         QTimer.singleShot(self.instruction_display_ms, self.show_stimulus)

#     def show_stimulus(self):
#         self.phase = "stimulus"
#         # FIXED: Show color bar only during stimulus
#         self.color_bar.show()
#         self.phase_duration = self.stim_ms
#         self.phase_elapsed = 0
#         self.progress.setValue(0)
#         self.set_background_color(QColor(0, 0, 0))
#         self.info_label.setText(f"Stimulus\n\nClass {self.current_class}")
#         self.color_bar.highlight_color(self.current_class)
#         self.progress_timer.start(100)
#         QTimer.singleShot(self.stim_ms, self.run_next_trial)

#     def update_progress(self):
#         self.phase_elapsed += 100
#         val = int((self.phase_elapsed / self.phase_duration) * 100)
#         self.progress.setValue(min(val, 100))
#         if val >= 100:
#             self.progress_timer.stop()

#     def set_background_color(self, color: QColor):
#         pal = self.palette()
#         pal.setColor(QPalette.Window, color)
#         self.setAutoFillBackground(True)
#         self.setPalette(pal)

#     def end_experiment(self):
#         self.phase = "done"
#         self.progress_timer.stop()
        
#         # FIXED: Hide color bar at end
#         self.color_bar.hide()
        
#         if self.unicorn_thread:
#             self.unicorn_thread.stop()
#             self.unicorn_thread.join(timeout=2)

#         self.log_file.close()
#         self.set_background_color(QColor(0, 0, 0))
#         self.info_label.setText("EEG data collection\ncompleted!")
#         self.progress.setValue(100)
#         print("CSV saved and experiment ended.")


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     gui = EEGTrialGUI(num_classes=5, trials_per_class=2, baseline_ms=1000, instruction_display_ms=2000, stim_ms=5000)
#     gui.show()
#     sys.exit(app.exec_())



# import sys, random, csv, threading, time
# from datetime import datetime
# from enum import Enum
# from PyQt5.QtCore import QTimer, Qt, QRect, QPoint
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QProgressBar, QHBoxLayout
# from PyQt5.QtGui import QColor, QPalette, QPainter, QPolygon
# import UnicornPy
# import numpy as np

# # Toggle EEG hardware integration
# USE_UNICORN = True


# # ========== DISPLAY MODE SYSTEM ==========
# class DisplayMode(Enum):
#     """Enum to define different visual display modes"""
#     BAR = "bar"
#     SHAPES = "shapes"


# class ColorDisplay(QWidget):
#     """Base class for color display widgets"""
#     def __init__(self, class_colors):
#         super().__init__()
#         self.class_colors = class_colors
#         self.highlight_index = None
#         self.setMinimumWidth(80)
#         self.setMaximumWidth(120)
    
#     def paintEvent(self, event):
#         """Each display mode implements its own painting logic"""
#         raise NotImplementedError("Subclasses must implement paintEvent()")
    
#     def highlight_color(self, class_index):
#         """Highlight one color (during stimulus)"""
#         self.highlight_index = class_index
#         self.update()
    
#     def reset_colors(self):
#         """Show all class colors (during instruction)"""
#         self.highlight_index = None
#         self.update()


# class ColorBar(ColorDisplay):
#     """Original horizontal bar display"""
#     def paintEvent(self, event):
#         painter = QPainter(self)
#         bar_width = self.width()
#         bar_height = self.height() // len(self.class_colors)

#         for i, (cls, color) in enumerate(self.class_colors.items()):
#             rect = QRect(0, i * bar_height, bar_width, bar_height)
#             if self.highlight_index is None:
#                 painter.fillRect(rect, color)
#             elif cls == self.highlight_index:
#                 painter.fillRect(rect, color)
#             else:
#                 painter.fillRect(rect, QColor(30, 30, 30))
#         painter.end()


# class ColorShapes(ColorDisplay):
#     """Display colored geometric shapes (circle, square, triangle, diamond, pentagon)"""
#     def __init__(self, class_colors):
#         super().__init__(class_colors)
#         self.setMinimumWidth(150)
#         self.setMaximumWidth(200)
    
#     def paintEvent(self, event):
#         painter = QPainter(self)
#         painter.setRenderHint(QPainter.Antialiasing)
        
#         total_shapes = len(self.class_colors)
#         shape_height = self.height() // total_shapes
#         center_x = self.width() // 2
#         shape_size = min(shape_height - 20, self.width() - 20)
        
#         # Define shape drawing functions
#         shape_types = [
#             self._draw_circle,
#             self._draw_square,
#             self._draw_triangle,
#             self._draw_diamond,
#             self._draw_pentagon
#         ]
        
#         for i, (cls, color) in enumerate(self.class_colors.items()):
#             center_y = i * shape_height + shape_height // 2
            
#             # Determine if this shape should be highlighted
#             if self.highlight_index is None:
#                 brush_color = color
#             elif cls == self.highlight_index:
#                 brush_color = color
#             else:
#                 brush_color = QColor(30, 30, 30)
            
#             painter.setBrush(brush_color)
#             painter.setPen(Qt.NoPen)
            
#             # Draw the appropriate shape for this class
#             shape_func = shape_types[i % len(shape_types)]
#             shape_func(painter, center_x, center_y, shape_size)
        
#         painter.end()
    
#     def _draw_circle(self, painter, x, y, size):
#         """Draw a circle"""
#         painter.drawEllipse(QPoint(x, y), size // 2, size // 2)
    
#     def _draw_square(self, painter, x, y, size):
#         """Draw a square"""
#         half = size // 2
#         rect = QRect(x - half, y - half, size, size)
#         painter.drawRect(rect)
    
#     def _draw_triangle(self, painter, x, y, size):
#         """Draw an upward-pointing triangle"""
#         half = size // 2
#         points = [
#             QPoint(x, y - half),
#             QPoint(x - half, y + half),
#             QPoint(x + half, y + half)
#         ]
#         painter.drawPolygon(QPolygon(points))
    
#     def _draw_diamond(self, painter, x, y, size):
#         """Draw a diamond (rotated square)"""
#         half = size // 2
#         points = [
#             QPoint(x, y - half),
#             QPoint(x + half, y),
#             QPoint(x, y + half),
#             QPoint(x - half, y)
#         ]
#         painter.drawPolygon(QPolygon(points))
    
#     def _draw_pentagon(self, painter, x, y, size):
#         """Draw a regular pentagon"""
#         import math
#         radius = size // 2
#         points = []
#         for i in range(5):
#             angle = math.radians(i * 72 - 90)
#             px = x + radius * math.cos(angle)
#             py = y + radius * math.sin(angle)
#             points.append(QPoint(int(px), int(py)))
#         painter.drawPolygon(QPolygon(points))


# def create_color_display(display_mode, class_colors):
#     """Factory function to create the appropriate display widget"""
#     if display_mode == DisplayMode.BAR or display_mode == "bar":
#         return ColorBar(class_colors)
#     elif display_mode == DisplayMode.SHAPES or display_mode == "shapes":
#         return ColorShapes(class_colors)
#     else:
#         raise ValueError(f"Unknown display mode: {display_mode}")


# # ========== EEG RECORDING THREADS ==========
# class UnicornRecorder(threading.Thread):
#     def __init__(self, gui_ref):
#         super().__init__(daemon=True)
#         self.gui = gui_ref
#         self.running = False
#         self.device = None
#         self.fs = 250
#         self.num_channels = None

#     def connect(self):
#         devices = UnicornPy.GetAvailableDevices(True)
#         if not devices:
#             raise RuntimeError("No Unicorn device found.")
        
#         print(f"Connecting to device: {devices[0]}")
#         self.device = UnicornPy.Unicorn(devices[0])
        
#         self.num_channels = self.device.GetNumberOfAcquiredChannels()
#         print(f"Device has {self.num_channels} channels")
#         print(f"Sample rate: {self.fs} Hz")
        
#         try:
#             self.device.StartAcquisition(False)
#             print("Acquisition started successfully.")
#         except Exception as e:
#             print("Failed to start acquisition:", e)
#             raise
        
#     def run(self):
#         if self.device is None:
#             print("No device connected.")
#             return

#         self.running = True
        
#         samples_per_read = 4
#         buffer_size = samples_per_read * self.num_channels * 4
        
#         print(f"Recording: {self.num_channels} channels, {samples_per_read} samples per read")
#         print(f"Buffer size: {buffer_size} bytes")
#         print("Starting data acquisition loop...")

#         total_samples = 0
#         last_flush_time = time.time()
        
#         time.sleep(0.5)
        
#         try:
#             while self.running:
#                 receive_buffer = bytearray(buffer_size)
#                 self.device.GetData(samples_per_read, receive_buffer, buffer_size)
#                 data = np.frombuffer(receive_buffer, dtype=np.float32)
#                 data = data.reshape((samples_per_read, self.num_channels))
                
#                 for sample in data:
#                     total_samples += 1
#                     trial = self.gui.current_trial
#                     cls = self.gui.current_class
#                     phase = getattr(self.gui, "phase", "idle")
#                     ts = datetime.now().strftime("%H:%M:%S.%f")
#                     row = [trial, cls, phase, ts] + sample.tolist()
#                     self.gui.csv_writer.writerow(row)
                
#                 current_time = time.time()
#                 if current_time - last_flush_time >= 1.0:
#                     self.gui.log_file.flush()
#                     last_flush_time = current_time
#                     print(f"✓ {total_samples} samples recorded ({total_samples/250:.1f}s)")
                
#         except KeyboardInterrupt:
#             print("Recording interrupted by user")
#         except Exception as e:
#             print(f"❌ Fatal error after {total_samples} samples: {e}")
#             import traceback
#             traceback.print_exc()

#         print(f"EEG recording stopped. Total: {total_samples} samples ({total_samples/250:.1f}s)")

#     def stop(self):
#         print("Stopping recorder thread...")
#         self.running = False
#         time.sleep(0.5)
#         if self.device:
#             try:
#                 self.device.StopAcquisition()
#                 print("Acquisition stopped.")
#             except Exception as ex:
#                 print(f"Stop error: {ex}")
#             try:
#                 del self.device
#                 self.device = None
#                 print("Device disconnected.")
#             except Exception as ex:
#                 print(f"Disconnect error: {ex}")


# class MockEEGRecorder(threading.Thread):
#     def __init__(self, gui_ref):
#         super().__init__(daemon=True)
#         self.gui = gui_ref
#         self.running = False
#         self.num_channels = 17

#     def run(self):
#         self.running = True
#         print("Running mock EEG recorder (17 channels)")
#         while self.running:
#             fake_data = [random.uniform(-100, 100) for _ in range(self.num_channels)]
#             ts = datetime.now().strftime("%H:%M:%S.%f")
#             trial = self.gui.current_trial
#             cls = self.gui.current_class
#             phase = getattr(self.gui, "phase", "idle")
#             self.gui.csv_writer.writerow([trial, cls, phase, ts] + fake_data)
#             self.gui.log_file.flush()
#             time.sleep(1 / 250)

#     def stop(self):
#         self.running = False


# # ========== MAIN EEG GUI ==========
# class EEGTrialGUI(QWidget):
#     def __init__(self, num_classes=5, trials_per_class=3, baseline_ms=3000, 
#                  instruction_display_ms=3000, stim_ms=3000, display_mode="bar"):
#         super().__init__()

#         self.num_classes = num_classes
#         self.trials_per_class = trials_per_class
#         self.baseline_ms = baseline_ms
#         self.instruction_display_ms = instruction_display_ms
#         self.stim_ms = stim_ms
#         self.display_mode = display_mode

#         color_pool = [
#             QColor(255, 0, 0),
#             QColor(255, 0, 0)
#             # QColor(0, 255, 0),
#             # QColor(0, 0, 255),
#             # QColor(255, 255, 0),
#             # QColor(255, 0, 255),
#         ]
#         self.class_colors = {i + 1: color_pool[i] for i in range(num_classes)}
#         self.color_names = {1: "Red",2: "Red"}
#                             #  2: "Green", 3: "Blue", 4: "Yellow", 5: "Magenta"}

#         self.trial_order = [c for c in range(1, num_classes + 1) for _ in range(trials_per_class)]
#         random.shuffle(self.trial_order)
#         self.total_trials = len(self.trial_order)
#         self.current_trial = 0
#         self.current_class = 0
#         self.phase = "idle"
#         self.is_running = False

#         self.setWindowTitle(f"EEG Data Collection GUI - {display_mode.upper()} Mode")
#         self.resize(1000, 700)

#         main_layout = QHBoxLayout()
#         left_layout = QVBoxLayout()
        
#         left_layout.addStretch(1)

#         self.info_label = QLabel("Press 'Start' to begin data collection")
#         self.info_label.setAlignment(Qt.AlignCenter)
#         self.info_label.setStyleSheet(
#             "font-size: 28px; font-weight: 600; color: white; "
#             "padding: 20px; background-color: rgba(0, 0, 0, 0);"
#         )
#         self.info_label.setWordWrap(True)
#         left_layout.addWidget(self.info_label)
        
#         left_layout.addSpacing(30)

#         self.progress = QProgressBar()
#         self.progress.setAlignment(Qt.AlignCenter)
#         self.progress.setStyleSheet(
#             "QProgressBar { border: 2px solid white; border-radius: 5px; "
#             "text-align: center; background-color: #1a1a1a; color: white; }"
#             "QProgressBar::chunk { background-color: #4CAF50; }"
#         )
#         self.progress.setMinimumHeight(30)
#         left_layout.addWidget(self.progress)
        
#         left_layout.addSpacing(30)

#         self.start_button = QPushButton("Start")
#         self.start_button.setStyleSheet(
#             "font-size: 18px; padding: 15px; background-color: #4CAF50; "
#             "color: white; border: none; border-radius: 5px;"
#         )
#         self.start_button.clicked.connect(self.start_experiment)
#         left_layout.addWidget(self.start_button)

#         left_layout.addStretch(2)

#         self.color_display = create_color_display(display_mode, self.class_colors)
#         self.color_display.hide()

#         main_layout.addLayout(left_layout, 5)
#         main_layout.addWidget(self.color_display, 1)
#         self.setLayout(main_layout)

#         self.set_background_color(QColor(0, 0, 0))

#         filename = f"EEG_{display_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
#         self.log_file = open(filename, "w", newline="")
#         self.csv_writer = csv.writer(self.log_file)
        
#         header = ["trial_index", "class_label", "phase", "timestamp"]
#         header += [f"Ch{i+1}" for i in range(17)]
#         self.csv_writer.writerow(header)

#         self.progress_timer = QTimer()
#         self.progress_timer.timeout.connect(self.update_progress)

#         if USE_UNICORN:
#             self.unicorn_thread = UnicornRecorder(self)
#         else:
#             self.unicorn_thread = MockEEGRecorder(self)

#     def start_experiment(self):
#         if self.is_running:
#             return
#         self.is_running = True
#         self.start_button.setEnabled(False)
#         self.info_label.setText("Connecting to EEG device...")
        
#         if USE_UNICORN:
#             try:
#                 self.unicorn_thread.connect()
#                 self.unicorn_thread.start()
#                 self.info_label.setText("Connected! Starting experiment...")
#             except Exception as e:
#                 print("EEG connection error:", e)
#                 self.info_label.setText(f"Connection failed: {e}")
#                 self.is_running = False
#                 self.start_button.setEnabled(True)
#                 return
#         else:
#             self.unicorn_thread.start()
#             print("Running GUI with mock EEG recorder.")
#             self.info_label.setText("Mock mode - Starting experiment...")
        
#         QTimer.singleShot(1000, self.run_next_trial)

#     def run_next_trial(self):
#         if self.current_trial >= self.total_trials:
#             self.end_experiment()
#             return
#         self.current_class = self.trial_order[self.current_trial]
#         self.current_trial += 1
#         self.show_baseline()

#     def show_baseline(self):
#         self.phase = "baseline"
#         self.color_display.hide()
#         self.phase_duration = self.baseline_ms
#         self.info_label.setText(f"Trial {self.current_trial}/{self.total_trials}\n\nBaseline")
#         self.phase_elapsed = 0
#         self.progress.setValue(0)
#         self.set_background_color(QColor(0, 0, 0))
#         self.color_display.reset_colors()
#         self.progress_timer.start(100)
#         QTimer.singleShot(self.baseline_ms, self.show_instruction)

#     def show_instruction(self):
#         self.phase = "instruction"
#         self.color_display.hide()
#         self.phase_duration = self.instruction_display_ms
#         color_name = self.color_names[self.current_class]
#         self.info_label.setText(f"Think about color:\n\n{color_name}")
#         self.phase_elapsed = 0
#         self.progress.setValue(0)
#         self.set_background_color(QColor(0, 0, 0))
#         self.color_display.reset_colors()
#         self.progress_timer.start(100)
#         QTimer.singleShot(self.instruction_display_ms, self.show_stimulus)

#     def show_stimulus(self):
#         self.phase = "stimulus"
#         self.color_display.show()
#         self.phase_duration = self.stim_ms
#         self.phase_elapsed = 0
#         self.progress.setValue(0)
#         self.set_background_color(QColor(0, 0, 0))
#         self.info_label.setText(f"Stimulus\n\nClass {self.current_class}")
#         self.color_display.highlight_color(self.current_class)
#         self.progress_timer.start(100)
#         QTimer.singleShot(self.stim_ms, self.run_next_trial)

#     def update_progress(self):
#         self.phase_elapsed += 100
#         val = int((self.phase_elapsed / self.phase_duration) * 100)
#         self.progress.setValue(min(val, 100))
#         if val >= 100:
#             self.progress_timer.stop()

#     def set_background_color(self, color: QColor):
#         pal = self.palette()
#         pal.setColor(QPalette.Window, color)
#         self.setAutoFillBackground(True)
#         self.setPalette(pal)

#     def end_experiment(self):
#         self.phase = "done"
#         self.progress_timer.stop()
#         self.color_display.hide()
        
#         if self.unicorn_thread:
#             self.unicorn_thread.stop()
#             self.unicorn_thread.join(timeout=2)

#         self.log_file.close()
#         self.set_background_color(QColor(0, 0, 0))
#         self.info_label.setText("EEG data collection\ncompleted!")
#         self.progress.setValue(100)
#         print("CSV saved and experiment ended.")


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
    
#     # Choose display mode: "bar" or "shapes"
#     gui = EEGTrialGUI(num_classes=2, trials_per_class=2, baseline_ms=1000, 
#                       instruction_display_ms=2000, stim_ms=5000, display_mode="bar")
    
#     gui.show()
#     sys.exit(app.exec_())


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