import pygame
import sys
import time
import random
import UnicornPy
import numpy as np
from pygame import gfxdraw
import argparse
import os
import glob
import threading
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from scipy.linalg import  eigh
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from scipy.signal import butter, resample, lfilter
import joblib
from scripts.train import TrainConfig, train, initialize_model
from scripts.online_infer import predict
# from ML_Model import ML_Model, FBCSP_OVR

# from EEGModel import EEGModel

class PlayerCar:
    def __init__(self):
        self.lane = 1
        self.image = pygame.image.load('./game/PlayerCar/Player.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (CAR_WIDTH*STEP, CAR_HEIGHT*STEP))
        self.rect =  self.image.get_rect()
        self.update_rect()

    def update_rect(self):
        self.rect.centerx = LANE_POSITION[self.lane]
        self.rect.bottom = SCREEN_HEIGHT - 20

    def move_left(self):
        if self.lane > 0:
            self.lane -= 1
            self.update_rect()

    def move_right(self):
        if self.lane < 2:
            self.lane += 1
            self.update_rect()

    def draw(self, screen):
        screen.blit(self.image, self.rect)

class ObstacleCar:
    def __init__(self, lane):
        self.lane = lane

        # Load random obstacle car sprite
        obstacle_folder = "./game/Obstacles"
        obstacle_files = [f for f in os.listdir(obstacle_folder) if f.endswith('.png')]
        random_obstacle = random.choice(obstacle_files)
        self.image = pygame.image.load(os.path.join(obstacle_folder, random_obstacle)).convert_alpha()
        self.image = pygame.transform.scale(self.image, (CAR_WIDTH*STEP, CAR_HEIGHT*STEP))
        self.rect = self.image.get_rect()
        self.rect.centerx = LANE_POSITION[self.lane]
        self.rect.bottom = -CAR_HEIGHT
        self.speed = CAR_SPEED

    def update(self):
        self.rect.y += self.speed
        return self.rect.top > SCREEN_HEIGHT    # Return TRUE if off the screen
    
    def draw(self, screen):
        #pygame.draw.rect(screen, self.color, self.rect)
        screen.blit(self.image, self.rect)




# Initialize game
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 1000
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Unicorn EEG Game')

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 120, 215)
GRAY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Game Constants
LANE_WIDTH = SCREEN_WIDTH // 3
CAR_WIDTH = 50
CAR_HEIGHT = 80
LANE_POSITION = [LANE_WIDTH//2, LANE_WIDTH + LANE_WIDTH//2, 2*(LANE_WIDTH)+LANE_WIDTH//2]
CAR_SPEED = 5
STEP = 2
OBSTACLE_SPAWN_RATE = 0.015
MIN_OBSTACLE_GAP = CAR_HEIGHT * 2
SPEED = 40

# Game States
STATE_SELECT_DEVICE = 0
STATE_COLLECT_DATA = 1
STATE_TRAIN_MODEL = 2
STATE_PLAY_GAME = 3
TRAIN_TRIALS = 75
TEST_TRIALS = 5
NUM_TRIALS = TRAIN_TRIALS + TEST_TRIALS
COUNTDOWN = 20

INSTRUCTION_DURATION = 3

#TODO: to store subject dependent data (with caliberaton)
# def get_data_file_path():
#     parser = argparse.ArgumentParser(description="EEG Game data file setup")
#     parser.add_argument('folder', nargs='?', default='./data', help='Optional folder path for data file')
#     parser.add_argument('username', nargs='?', default=None, help='Username to create data file for (optional)')

#     args = parser.parse_args()

#     folder = args.folder
#     username = args.username

#     # Ensure folder exists
#     if not os.path.exists(folder):
#         os.makedirs(folder)

#     if username:
#         username = username.lower()

#         # Search existing files for latest run number
#         pattern = os.path.join(folder, f"{username}_run_*.csv")
#         existing_files = glob.glob(pattern)
        
#         run_numbers = []
#         for f in existing_files:
#             basename = os.path.basename(f)
#             try:
#                 num_str = basename.split('_run_')[1].split('.csv')[0]
#                 run_numbers.append(int(num_str))
#             except (IndexError, ValueError):
#                 continue

#         next_run = max(run_numbers) + 1 if run_numbers else 0
#         filename = f"{username}_run_{next_run}.csv"

#     else:
#         # Use anonymous with current date-time
#         now_str = time.strftime("%Y%m%d_%H%M%S")
#         filename = f"anonymous_{now_str}.csv"

#     file_path = os.path.join(folder, filename)
#     return file_path

# DATA_FILE_LOCATION = get_data_file_path()

# def get_data_files():
#     parser = argparse.ArgumentParser(description="EEG Game data files setup")
#     parser.add_argument('train_file', nargs='?', default=None, help='Optional folder path for data file')
#     parser.add_argument('test_file', nargs='?', default=None, help='Username to create data file for (optional)')

#     args = parser.parse_args()

#     train_file = args.train_file
#     test_file = args.test_file
#     if train_file is None:
#         raise argparse.ArgumentTypeError(f"Please provide training data file")
#     if not os.path.exists(train_file):
#             raise argparse.ArgumentTypeError(f"Training data file not found at location: {train_file}")
#     if test_file is not None:
#         if not os.path.exists(test_file):
#             raise argparse.ArgumentTypeError(f"Testing data file not found at location: {test_file}")
#     if test_file is None:
#         print("Testing data file not provided hence using training file instead.")
    
#     print("Successfully located training and testing files")
#     return train_file, test_file if test_file is not None else  train_file 

DATA_STORAGE_FILE = '../data/shivansh_run_3.csv'
TRAINING_DATA_FILE = DATA_STORAGE_FILE
TESTING_DATA_FILE = '../data/shivansh_run_0.csv'

# Testing log to be saved >
TEST_LOG_FILE = f"{os.path.splitext(TRAINING_DATA_FILE)[0]}_test_log_with_obsticles_dry_run.npz"

MODEL_PATH = "./outputs/trained_models/shivansh_run_3_best.pth"

def prepare_image(img):
    return pygame.transform.scale(pygame.image.load(img).convert_alpha(), (100,100))
class UnicornDeviceSelector:
    def __init__(self):
        self.font = pygame.font.SysFont('Arial', 32)
        self.small_font = pygame.font.SysFont('Arial', 24)
        self.devices = []
        self.selected_devices = None
        self.state = STATE_SELECT_DEVICE # Don't change this configuration!
        self.train_data_file = TRAINING_DATA_FILE
        self.data_file = TESTING_DATA_FILE
        self.test_data_file = TESTING_DATA_FILE
        self.current_phase = 'baseline'
        self.phases = {
            "baseline": 1.0,      # 1s
            "instruction": 1.0,   # 1s
            "mi": 4.0             # 4s
        }
        self.waiting_input = False
        self.holding_key = False
        self.trial_start_time = time.time()
        self.key_press_time = 0
        self.data_buffer = deque(maxlen=250*30)  # Buffer for 30 seconds of data at 250Hz
        self.lock = threading.Lock()

        self.model = None
        self.scaler = None
        self.training_complete = False
        self.training_progress = 0
        self.last_prediction_time = 0
        self.prediction_interval = 1 # Predict every 2s
        self.readtime_eeg_buffer = deque(maxlen=250*20)
        self.data_collection_mode = 'train'
        self.training_started = False
        # Master dictionary of calibration instructions
        self.calibration_instructions = {
            "left_arm": {
                "id":"left_arm",
                "key": pygame.K_z,
                "image": pygame.transform.scale(pygame.image.load('./game/Hand/left_hand.png').convert_alpha(), (100,100)),
                "instruction": {"main": "Focus on the object on your LEFT", "status": "Get ready to think about movement"},
                "mi": {"main": "Think about grabbing object on your LEFT.", "status": lambda x: f"Keep thinking about LEFT hand movement for {x:.1f}s"}
            },
            "right_arm": {
                "id":"right_arm",
                "key": pygame.K_RIGHT,
                "image": pygame.transform.scale(pygame.image.load('./game/Hand/right_hand.png').convert_alpha(), (100,100)),
                "instruction": {"main": "Focus on the object on your RIGHT", "status": "Get ready to think about movement"},
                "mi": {"main": "Think about grabbing object on your RIGHT.", "status": lambda x: f"Keep thinking about RIGHT hand movement for {x:.1f}s"}
            },
            "still": {
                "id":"still",
                "key": None,
                "image": pygame.transform.scale(pygame.image.load("./game/Hand/Still.png").convert_alpha(), (100,100)),
                "instruction": {"main": "Focus on the screen", "status": "Remain still"},
                "mi": {"main": "Remain still", "status": lambda x: f"Hold still ... ({x:.1f})"}
            },
            "left_leg": {
                "id":"left_leg",
                "key": pygame.K_a,
                "image": pygame.transform.scale(pygame.image.load('./game/Leg/left_leg.png').convert_alpha(), (100,100)),
                "instruction": {"main": "Focus on your LEFT FOOT", "status": "Get ready to think about movement"},
                "mi": {"main": "Think about lifting up your LEFT FOOT.", "status": lambda x: f"Keep thinking about LEFT FOOT movement for {x:.1f}s"}
            },
            "right_leg": {
                "id":"right_leg",
                "key": pygame.K_d,
                "image": pygame.transform.scale(pygame.image.load('./game/Leg/right_leg.png').convert_alpha(), (100,100)),
                "instruction": {"main": "Focus on your RIGHT FOOT", "status": "Get ready to think about movement"},
                "mi": {"main": "Think about lifting up your RIGHT FOOT.", "status": lambda x: f"Keep thinking about RIGHT FOOT movement for {x:.1f}s"}
            }
        }

        self.instruction_info = {"train": [], "test": []}
        self.prediction_thread = None
        self.prediction_active = False
        self.prediction_lock = threading.Lock()
        self.latest_prediction = "still"

        for _ in range(TRAIN_TRIALS):
            for k, v in self.calibration_instructions.items():
                self.instruction_info["train"].append(v)

        for _ in range(TEST_TRIALS):
            for k, v in self.calibration_instructions.items():
                self.instruction_info["test"].append(v)

        # Start with training
        self.instructions = self.instruction_info["train"]
        random.shuffle(self.instructions)
        
        
        self.current_instruction = 0
        self.instruction_start_time = 0
        self.waiting_input = False
        self.correct_input_received = False
        self.recording_duration = 3  # seconds per movement

        self.arrow_images = {
            pygame.K_z:pygame.transform.scale(pygame.image.load('./game/Hand/left_hand.png').convert_alpha(), (100,100)),
            pygame.K_RIGHT:pygame.transform.scale(pygame.image.load('./game/Hand/Right_hand.png').convert_alpha(), (100,100)),
            "Still":pygame.transform.scale(pygame.image.load("./game/Hand/Still.png").convert_alpha(), (100,100))
        }

    def get_available_devices(self):
        try:
            self.devices = UnicornPy.GetAvailableDevices(True)
            if len(self.devices) <= 0:
                raise Exception("No Device Found. Pair a device")
        except Exception as e:
            print(f"Error getting devices: {e}")
            self.devices = []

    def draw_device_selection(self, screen):
        screen.fill(BLACK)
        title = self.font.render("Select Unicorn Device", True, WHITE)
        screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 50))

        if not self.devices:
            error_text = self.small_font.render("No devices available. Pair something", True, RED)
            screen.blit(error_text, (SCREEN_HEIGHT//2 - error_text.get_width()//2, 150))
            return 
        
        for i, device in enumerate(self.devices):
            y_pos = 150 + i * 50
            color = WHITE
            if i == self.selected_devices:
                color = BLUE 
            
            device_text = self.small_font.render(f"{i}: {device}", True, color)
            screen.blit(device_text, (SCREEN_WIDTH//2 - device_text.get_width()//2, y_pos))

        instruction = self.small_font.render("Use UP/DOWN arrows to select, C to confirm", True, WHITE)
        screen.blit(instruction, (SCREEN_WIDTH//2 - instruction.get_width()//2, SCREEN_HEIGHT - 100))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                # Device Selection State
                if self.state == STATE_SELECT_DEVICE:
                    if event.key == pygame.K_UP and self.selected_devices is not None:
                        self.selected_devices = max(0, self.selected_devices-1)
                    elif event.key == pygame.K_DOWN and self.selected_devices is not None:
                        self.selected_devices = min(len(self.devices)-1, self.selected_devices+1)
                    elif event.key == pygame.K_c and self.selected_devices is not None:
                        # self.state = STATE_COLLECT_DATA
                        self.start_data_acquisition()
                        # self.current_phase = "baseline"
                        # self.waiting_input = False
                        # self.holding_key = False
                        # self.trial_start_time = time.time()
                        self.state = STATE_TRAIN_MODEL
                
                # EEG Data Collection State
                elif self.state == STATE_COLLECT_DATA:
                    if self.current_phase == "reach" and self.waiting_input:
                        self.holding_key = True
                        self.key_press_time = time.time()
                        self.current_phase = "hold"
                        # required_key = self.instructions[self.current_instruction][1]
                        
                        # if required_key is not None and event.key == required_key:
                        #     self.holding_key = True
                        #     self.key_press_time = time.time()
                        #     self.current_phase = "hold"
                    
                    elif event.key == pygame.K_r:
                        self.reset_data_collection()
                
                # Game State
                elif self.state == STATE_PLAY_GAME:
                    if event.key == pygame.K_r:
                        self.reset_game()
                    if event.key == pygame.K_m: # Manual override
                        if event.key == pygame.K_z:
                            self.player.move_left()
                        elif event.key == pygame.K_RIGHT:
                            self.player.move_right()

    def format_timestamp(self, timestamp):
        hrs = int((timestamp // 3600) % 24)-4
        mins = int((timestamp % 3600) // 60)
        sec = int(timestamp % 60)
        milli = int((timestamp - int(timestamp)) * 1000)
        return f"{hrs:02d}:{mins:02d}:{sec:02d}.{milli:04d}"

    def start_data_acquisition(self):
        try:
            print(f"Connecting to device {self.devices[self.selected_devices]}")
            self.device = UnicornPy.Unicorn(self.devices[self.selected_devices])
            print("Connected Successfully")

            # Get device configuration
            #self.sampling_rate = self.device.GetSamplingRate()  # Typically 250Hz
            self.sampling_rate = 250
            self.number_of_channels = self.device.GetNumberOfAcquiredChannels()
            
            # Calculate buffer size for about 60Hz updates (4 samples at 250Hz)
            self.frame_length = max(1, int(self.sampling_rate / 60))
            self.receive_buffer = bytearray(self.frame_length * self.number_of_channels * 4)
            
            # Give 10 s for the device to be stable
            self.display_wait_countdown(COUNTDOWN)

            # Open data file
            self.file = open(self.data_file, 'wb')
            
            # # Start acquisition thread
            # self.acquisition_active = True
            # self.acquisition_thread = threading.Thread(target=self.data_acquisition_thread)
            # self.acquisition_thread.start()
            
        except Exception as e:
            print(f"Error starting acquisition: {e}")
            self.state = STATE_SELECT_DEVICE

    def display_wait_countdown(self, counter):
        start_time = time.time()
        screen = pygame.display.get_surface()

        while time.time() - start_time < counter:
            remaining = counter - int(time.time() - start_time)

            # Clear screen
            screen.fill(BLACK)

            # Display countdown message
            title = self.font.render("Device Stabilization", True, BLUE)
            screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, SCREEN_HEIGHT//2-50))

            # Display countdonw
            countdown = self.font.render(f"Starting in : {remaining} seconds", True, GREEN)
            screen.blit(countdown, (SCREEN_WIDTH//2 - countdown.get_width()//2, SCREEN_HEIGHT//2))

            # Instruction display
            instruction = self.small_font.render("Please remain still during stabilization", True, WHITE)
            screen.blit(instruction, (SCREEN_WIDTH//2 - instruction.get_width()//2, SCREEN_HEIGHT//2 + 50))
            pygame.display.flip()

            # Handle events so window dont freeze
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return False
                
            time.sleep(0.1)

        return True

    def data_acquisition_thread(self):
    # Thread that continuously reads data from the Unicorn device
        try:
            self.device.StartAcquisition(False)
            last_write_time = time.time()  # Initialize the variable here
            
            while self.acquisition_active:

                # Check if there is still instructions to process
                if self.state != STATE_PLAY_GAME:
                    if self.current_instruction >= len(self.instructions) and self.data_collection_mode=='test':
                        break

                    # Get current timestamp and phase info
                    timestamp = self.format_timestamp(time.time())
                    # direction = 'left' if self.instructions[self.current_instruction][1] == pygame.K_z else "right"

                    if self.current_instruction < len(self.instructions):
                        #direction = "left" if self.instructions[self.current_instruction][1] == pygame.K_z else "right"
                        direction = self.instructions[self.current_instruction].get("id")
                        phase = self.current_phase
                    mode = self.data_collection_mode
                    # Read fixed number of samples
                    samples_to_read = 4  # About 60Hz at 250Hz sampling rate
                    buffer_size = samples_to_read * self.number_of_channels * 4
                    receive_buffer = bytearray(buffer_size)
                    self.device.GetData(samples_to_read, receive_buffer, buffer_size)
                    data = np.frombuffer(receive_buffer, dtype=np.float32)
                    data = data.reshape((samples_to_read, self.number_of_channels))
                    
                    # Store data
                    with self.lock:
                        for sample in data:
                            self.readtime_eeg_buffer.append(sample[:8])

                            # Collection phase
                            if self.state == STATE_COLLECT_DATA:
                                self.data_buffer.append({
                                    'timestamp': timestamp,
                                    'direction': direction,
                                    'phase': phase,
                                    'data': sample,
                                    "mode" : mode
                                })
                            
                    # Write to file every second
                    if self.state == STATE_COLLECT_DATA and time.time() - last_write_time > 1.0:
                        self.write_buffered_data()
                        last_write_time = time.time()      
                else:
                    samples_to_read = 4  # About 60Hz at 250Hz sampling rate
                    buffer_size = samples_to_read * self.number_of_channels * 4
                    receive_buffer = bytearray(buffer_size)
                    self.device.GetData(samples_to_read, receive_buffer, buffer_size)
                    data = np.frombuffer(receive_buffer, dtype=np.float32)
                    data = data.reshape((samples_to_read, self.number_of_channels))                        
                    with self.lock:
                        for sample in data:
                            self.readtime_eeg_buffer.append(sample[:8])
        except Exception as e:
            print(f"Error in acquisition thread: {e}")
        finally:
            self.device.StopAcquisition()

    def write_buffered_data(self):
        """Write buffered data to file"""
        if self.state == STATE_COLLECT_DATA:
            with self.lock:
                if len(self.data_buffer) == 0:
                    return
                    
                with open(self.data_file, 'ab') as f:
                    for entry in self.data_buffer:
                        row = f"{entry['timestamp']},{entry['direction']},{entry['phase']},{entry['mode']}" + \
                            ",".join([f"{x:.3f}" for x in entry['data']]) + "\n"
                        f.write(row.encode())
                
                self.data_buffer.clear()
        pass

    def update_data_acquisition(self):
        if self.current_instruction >= len(self.instructions):
            if self.data_collection_mode == "test":
                self.stop_data_acquisition()
                self.state = STATE_TRAIN_MODEL
                self.start_model_training()
                return True
            else:
                self.data_collection_mode = "test"
                self.instructions = self.instruction_info["test"]
                random.shuffle(self.instructions)
                self.current_instruction = 0
                return False

        current_time = time.time()
        elapsed = current_time - self.trial_start_time

        if self.current_phase == "baseline" and elapsed >= 1.0:
            self.current_phase = "instruction"
            self.trial_start_time = current_time

        elif self.current_phase == "instruction" and elapsed >= 1.0:
            self.current_phase = "mi"
            self.trial_start_time = current_time

        elif self.current_phase == "mi" and elapsed >= 4.0:
            self.current_instruction += 1
            self.current_phase = "baseline"
            self.trial_start_time = current_time

        return False
    
    def stop_data_acquisition(self):
        self.acquisition_active = False
        if hasattr(self, 'acquisition_thread'):
            self.acquisition_thread.join(timeout=1.0)
        self.write_buffered_data()
        if hasattr(self, 'file'):
            self.file.close()
        print("Data acquisition completed successfully")

    def start_model_training(self):
        self.training_started = True
        self.training_thread = threading.Thread(target=self.train_model)
        self.training_thread.start()
        self.training_progress = 0

    def train_model(self):
        # try:
            # folder_path = "D:\\Parthan\\Real-Time EEG Control\\Left_Right_Game\\"   
            # file_path = folder_path + self.data_file
            # data = pd.read_csv(file_path, header=None)
            # print("Processing Data")
            # data = data.iloc[:,0:11]
            # # Initialize and process data
            # self.training_progress = 0.25
            # ml_model = ML_Model(data, self.data_file)
            # ml_model.process()

            # # Store the trained model
            # self.svm_model = ml_model
            # self.training_progress = 0.75

            # # Evaluation

            # self.training_progress = 1.0
            # self.training_complete = True

            # # Move to game
            # time.sleep(2)
            # self.state = STATE_PLAY_GAME

            # folder_path = ""
            # file_path = folder_path + self.data_file
            print("Processing Data")
            
            self.training_progress = 0.25
            
            # ml_model = EEGModel()
            # ml_model.train(self.train_data_file)
            model, is_trained = initialize_model(MODEL_PATH)
            if is_trained:
                print(f"Loaded trained model from:{MODEL_PATH}")
                self.label_map = MODEL_PATH.replace("_best.pth", "_label_map.json")
            else:
                args = TrainConfig(
                    csv=self.train_data_file,
                    batch=16,
                    epochs=50,
                    seed=42,
                    lr = 0.0001
                )
                model, _, _, self.label_map = train(model, args)
                print("Model Trained")
                # Store the trained model
            self.svm_model = model
            self.training_progress = 0.75

            # Simulate loading delay
            time.sleep(2)

            # Evaluation

            self.training_progress = 1.0
            self.training_complete = True

            # Move to game
            self.display_wait_countdown(COUNTDOWN)
            time.sleep(2)
            self.state = STATE_PLAY_GAME


        # except Exception as e:
        #     print(f"Error training Model: {e}")
        #     self.training_complete = False
        
    def draw_data_collection(self, screen):
        screen.fill(BLACK)
        
        total_phases = len(self.instruction_info['train']) + len(self.instruction_info['test'])
        completed = self.current_instruction + (len(self.instruction_info['train']) if self.data_collection_mode=="test" else 0)

        # Progress bar
        progress = completed / total_phases
        pygame.draw.rect(screen, WHITE, (100, SCREEN_HEIGHT//2 - 25, SCREEN_WIDTH-200, 50), 1)
        pygame.draw.rect(screen, GREEN, (100, SCREEN_HEIGHT//2 - 25, (SCREEN_WIDTH-200)*progress, 50))
        
        if self.current_instruction < len(self.instructions):
            trial = self.instructions[self.current_instruction]

            if self.current_phase == "baseline":
                instruction = "Remain still"
                remaining = max(0, 1.0 - (time.time() - self.trial_start_time))
                status = f"Baseline recording... {remaining:.1f}s"

            elif self.current_phase == "instruction":
                instruction = trial["instruction"]["main"]
                status = trial["instruction"]["status"]

            elif self.current_phase == "mi":
                remaining = max(0, 4.0 - (time.time() - self.trial_start_time))
                instruction = trial["mi"]["main"]
                status = trial["mi"]["status"](remaining)
                screen.blit(trial["image"], (SCREEN_WIDTH//2 - trial["image"].get_width()//2, SCREEN_HEIGHT//2 + 40))

            instruction_text = self.font.render(instruction, True, WHITE)
            screen.blit(instruction_text, (SCREEN_WIDTH//2 - instruction_text.get_width()//2, SCREEN_HEIGHT//2 - 100))

            status_text = self.small_font.render(status, True, WHITE)
            screen.blit(status_text, (SCREEN_WIDTH//2 - status_text.get_width()//2, SCREEN_HEIGHT//2 + 150))
        
        # percent = self.small_font.render(f"{int(progress * 100)}%", True, WHITE)
        # screen.blit(percent, (SCREEN_WIDTH//2 - percent.get_width()//2, SCREEN_HEIGHT//2 + 200))

    def draw_model_training(self, screen):
        screen.fill(BLACK)
        title = self.font.render("Training ML Model", True, WHITE)
        screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 100))

        # Progress bar
        pygame.draw.rect(screen, WHITE, (100, SCREEN_HEIGHT//2 - 25, SCREEN_WIDTH-200, 50), 1)
        pygame.draw.rect(screen, RED, (100, SCREEN_HEIGHT//2 - 25, (SCREEN_WIDTH-200)*self.training_progress, 50))

        status = ""
        if self.training_progress < 0.25:
            status = "Preprocessing data"
        elif self.training_progress < 0.75:
            status = "Training Model"
        else:
            status = "Training Complete! Starting Game ..."

        status_text = self.small_font.render(status, True, WHITE)
        screen.blit(status_text, (SCREEN_WIDTH//2 - status_text.get_width()//2, SCREEN_HEIGHT//2 + 50))

        percent = self.small_font.render(f"{int(self.training_progress * 100)}%", True, WHITE)
        screen.blit(percent, (SCREEN_WIDTH//2 - percent.get_width()//2, SCREEN_HEIGHT//2+100))

    def get_live_prediction(self):
        print("Inside")
        if len(self.readtime_eeg_buffer) < 250:
            return "still", []
        
        eeg_window = np.array(self.readtime_eeg_buffer, dtype=np.float32)[-250:, :8]

        # eeg_window = np.nan_to_num(eeg_window, nan=0.0, posinf=0.0, neginf=0.0)
        # eeg_window = eeg_window.T
        # print("shape", eeg_window.shape)
        # try:
        pred_class, pred_prob, = predict(self.svm_model, eeg_window,self.label_map)
        
        
        return pred_class, eeg_window
        # except Exception as e:
        #     print(f"Live Prediction Error {e}")
        #     return "still"

    def start_prediction_thread(self):
            if self.prediction_thread is None or not self.prediction_thread.is_alive():
                self.prediction_active = True
                self.prediction_thread = threading.Thread(target=self.prediction_loop, daemon=True)
                self.prediction_thread.start()
    def stop_prediction_thread(self):
        self.prediction_active = False
        if self.prediction_thread is not None:
            self.prediction_thread.join(timeout=1.0)
    
    def prediction_loop(self):
        while self.prediction_active:
            if time.time() - self.last_prediction_time > self.prediction_interval:
                self.last_prediction_time = time.time()
                pred, eeg_window = self.get_live_prediction()
                with self.prediction_lock:
                    self.latest_prediction = pred
            time.sleep(0.05)
class EEGGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.test_log_file = TEST_LOG_FILE
        pygame.display.set_caption("Gone in 30 seconds")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.reset_game()
        self.experiment_start_time = time.time()
        # self.exp_instruction = ["LEFT", "STILL", "RIGHT", "LEFT", "STILL", "RIGHT", "LEFT", "LEFT", "RIGHT", "STILL", "RIGHT", "LEFT", "STILL", "RIGHT", "LEFT", "LEFT", "STILL", "RIGHT"]
        self.unicorn_selector = UnicornDeviceSelector()
        self.unicorn_selector.get_available_devices()
        self.unicorn_selector.selected_devices = None
        # self.actual_text = "LEFT"
        # self.last_actual_text_change_time = time.time()

        self.exp_instructions = ["STILL", "RIGHT", "LEFT", "LEFT", "RIGHT", "STILL", "LEFT", "RIGHT", "STILL"]
        self.curr_instr = self.exp_instructions.pop(0)
        self.instr_start_time = time.time()
        self.spawned_this_instr = False
        self.prev_blocked_lanes = None


        if self.unicorn_selector.devices:
            self.unicorn_selector.selected_devices = 0

    def instruction_generator(self):
        lanes = [0,1,2]
        moves = ["LEFT", "STILL", "RIGHT"]

        current_lane = 1
        instructions = []
        lane_history = [current_lane]

        for _ in range(20):
            valid_moves = []
            if current_lane > 0:
                valid_moves.append("LEFT")
            valid_moves.append("STILL")
            if current_lane < 2:
                valid_moves.append("RIGHT")

            chosen_move = random.choice(valid_moves)
            instructions.append(chosen_move)

            if chosen_move == "LEFT":
                current_lane -= 1
            elif chosen_move == "RIGHT":
                current_lane += 1

            lane_history.append(current_lane)
        return instructions
        
    def reset_game(self):
        print("Resetting Game....")
        self.unicorn_selector.stop_prediction_thread()
        self.player = PlayerCar()
        self.obstacles = []
        self.game_time = 30
        self.start_time = time.time()
        self.game_over = False
        self.won = False
        self.last_prediction = "still"


        
    def run(self):
        if os.path.exists(self.unicorn_selector.data_file):
            overwrite = input("[Warning] There is already a file exists at path:{self.unicorn_selector.data_file}!\nDo you want to overwrite? y/n")
            if overwrite.lower()!='y':
                print("Overwrite disapproved!\nExiting pipeline...")
                return
        while True:
            if self.unicorn_selector.state == STATE_SELECT_DEVICE:
                self.unicorn_selector.handle_events()
                self.unicorn_selector.draw_device_selection(self.screen)
            elif self.unicorn_selector.state == STATE_COLLECT_DATA:
                pass
                self.unicorn_selector.handle_events()
                if self.unicorn_selector.update_data_acquisition():
                    self.reset_game()
                self.unicorn_selector.draw_data_collection(self.screen)
            
            elif self.unicorn_selector.state == STATE_TRAIN_MODEL:
                if not self.unicorn_selector.training_started:
                    self.unicorn_selector.training_started = True
                    self.unicorn_selector.start_model_training()
                self.unicorn_selector.handle_events()
                self.unicorn_selector.draw_model_training(self.screen)

            elif self.unicorn_selector.state == STATE_PLAY_GAME:
                if not hasattr(self.unicorn_selector, "acquisition_active") or not self.unicorn_selector.acquisition_active:
                    self.unicorn_selector.acquisition_active = True
                    self.unicorn_selector.acquisition_thread = threading.Thread(target=self.unicorn_selector.data_acquisition_thread)
                    self.unicorn_selector.acquisition_thread.start()
                self.handle_game_events()
                self.update()
                self.draw()

            pygame.display.flip()
            self.clock.tick(SPEED)

    def handle_game_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and (self.game_over or self.won):
                    self.reset_game()

                if event.key == pygame.K_ESCAPE:
                    self.unicorn_selector.state = STATE_SELECT_DEVICE

                if not self.game_over and not self.won:
                    if event.key == pygame.K_m: # Manual override 
                        if event.key == pygame.K_z:
                            self.player.move_left()

                        if event.key == pygame.K_RIGHT:
                            self.player.move_right()
    def save_eeg_npz(self, eeg_window, timestamp, actual, predicted):
        # Append data if file exists
        if os.path.exists(self.test_log_file):
            data = np.load(self.test_log_file, allow_pickle=True)
            eegs = data["eegs"].tolist()
            timestamps = data["timestamps"].tolist()
            actuals = data["true_value"].tolist()
            predictions = data["predicted_value"].tolist()
        else:
            eegs, timestamps, actuals, predictions = [], [], [], []

        eegs.append(eeg_window)
        timestamps.append(timestamp)
        actuals.append(actual)
        predictions.append(predicted)

        np.savez(self.test_log_file, eegs=eegs, timestamps=timestamps, true_value=actuals, predicted_value=predictions)
        return self.test_log_file
    
    def update(self):
        if self.game_over or self.won:
            return 0
        
        if time.time() - self.unicorn_selector.last_prediction_time > self.unicorn_selector.prediction_interval:
            print("Here")
            self.unicorn_selector.last_prediction_time = time.time()
            # prediction, eeg_window = self.unicorn_selector.get_live_prediction()
            # if len(eeg_window) > 0:
            #     self.save_eeg_npz(eeg_window, time.time(), self.actual_text.lower(), prediction.lower())

            # self.last_prediction = prediction
            with self.unicorn_selector.prediction_lock:
                prediction = self.unicorn_selector.latest_prediction

            if prediction == "left":
                self.player.move_left()
            elif prediction == "right":
                self.player.move_right()

        
        current_time = time.time()
        elapsed = current_time - self.start_time
        remaining = max(0, self.game_time - elapsed)

        if not self.game_over:
            self.spawn_obstacles_instructions()

            # Update Obstacles
            for obstacle in self.obstacles[:]:
                if obstacle.update():
                    self.obstacles.remove(obstacle)

            # Check collision
            for obstacle in self.obstacles:
                if self.player.rect.colliderect(obstacle.rect):
                    self.game_over = True
                    self.final_time = elapsed
                    break

            # if not self.game_over and remaining <= 0:
            #     self.won = True
            #     self.final_time = self.game_time

    # def spawn_obstacles(self):
    #     # Spawn obstacles
    #     if random.random() < OBSTACLE_SPAWN_RATE:
    #         available_lanes = [0, 1, 2]
    #         clear_lanes = [lane for lane in available_lanes if self.is_lane_clear(lane)]

    #         if len(clear_lanes) >= 2:
    #             if len(clear_lanes) > 2 or self.player.lane not in clear_lanes:
    #                 available_lanes = [lane for lane in clear_lanes if lane != self.player.lane]
    #                 if not available_lanes:
    #                     available_lanes = clear_lanes.copy()
    #                 lane = random.choice(clear_lanes)
    #                 self.obstacles.append(ObstacleCar(lane))
    
    def _lanes_to_block_for_instr(self, instr):
        lanes = [0, 1, 2]

        if instr == "LEFT":
            target = max(0, self.player.lane - 1)
            lanes_to_block = [l for l in lanes if l != target]
            self.prev_blocked_lanes = lanes_to_block
            return lanes_to_block
        
        if instr == "RIGHT":
            target = min(2, self.player.lane + 1)
            lanes_to_block = [l for l in lanes if l != target]
            self.prev_blocked_lanes = lanes_to_block
            return lanes_to_block
        
        if instr == "STILL":
            return [l for l in lanes if l != self.player.lane]

    def spawn_obstacles_instructions(self):
        if self.curr_instr is None:
            return 
        
        lanes_to_block = self._lanes_to_block_for_instr(self.curr_instr)

        if not self.spawned_this_instr:
            spawned = False
            for lane in lanes_to_block:
                if self.is_lane_clear(lane):
                    self.obstacles.append(ObstacleCar(lane))
                    spawned = True

            if spawned:
                self.spawned_this_instr = True
            self.instr_start_time = time.time()


        now = time.time()
        if now - self.instr_start_time >= INSTRUCTION_DURATION:
            if self.exp_instructions:
                self.curr_instr = self.exp_instructions.pop(0)
                self.instr_start_time = now
                self.spawned_this_instr = False
            else:
                self.curr_instr = None

    def is_lane_clear(self, lane):
        for obstacle in self.obstacles:
            if obstacle.lane == lane and obstacle.rect.top < MIN_OBSTACLE_GAP:
                return False
        return True
    
    def draw_road(self):
        self.screen.fill(GRAY)
        dash_length = 20
        gap_length = 15
        for y in range(0, SCREEN_HEIGHT, dash_length + gap_length):
            pygame.draw.line(self.screen, WHITE, (LANE_WIDTH, y), (LANE_WIDTH, y+dash_length), 2)
        
        for y in range(0, SCREEN_HEIGHT, dash_length + gap_length):
            pygame.draw.line(self.screen, WHITE, (2*LANE_WIDTH, y), (2*LANE_WIDTH, y+dash_length), 2)
    
    
    # def show_actual_text(self):
    #     instructions  =['LEFT', 'STILL', 'RIGHT']
    #     if len(self.exp_instruction) == 0:
    #         pygame.quit()
    #         sys.exit(1)
    #     if not self.actual_text:
    #         self.actual_text = instructions[0]
    #     if self.last_actual_text_change_time is not None and (time.time() - self.last_actual_text_change_time) > 5:
    #         self.last_actual_text_change_time = time.time()
    #         self.actual_text = self.exp_instruction.pop()
        
        # return self.actual_text 
    def draw(self):
        self.draw_road()

        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
        self.player.draw(self.screen)

        if self.game_over:
            time_survived = min(self.final_time, self.game_time)
            game_over_text = self.font.render(f"SURVIVED {int(time_survived)}s!! Press R to restart", True, WHITE)
            self.screen.blit(game_over_text, (SCREEN_WIDTH//2 - game_over_text.get_width()//2, SCREEN_HEIGHT//2))
        elif self.won:
                win_text = self.font.render("SURVIVED!! Press R to restart", True, WHITE)
                self.screen.blit(win_text, (SCREEN_WIDTH//2 - win_text.get_width()//2, SCREEN_HEIGHT//2))
        else:
            # remaining = max(0,  self.start_time - time.time())
            # timer_text = self.font.render(f"Time: {int(remaining)}s", True, WHITE)
            # self.screen.blit(timer_text, (10,10))

            prediction_text = self.font.render(f"Prediction: {self.last_prediction.upper()}", True, GREEN)
            self.screen.blit(prediction_text, (10, 50))
            
            # actual_text = self.font.render(f"Think: {self.show_actual_text().upper()}", True, RED)
            # self.screen.blit(actual_text, (10, 100))

            # Show control mode
            mode_text =self.font.render("EEG Control Active", True, WHITE)
            self.screen.blit(mode_text, (SCREEN_WIDTH//2 - mode_text.get_width()//2, 10))

        pygame.display.flip()


if __name__ == "__main__":
    game = EEGGame()
    game.run()