# ============================================================================
# Hardware Configuration - Shared Config for New Components
# ============================================================================
# Kinova arm setup: mirrors OpenCV/track_ball_kinova_test_fixed.py
# EEG headset setup: mirrors dataCollection.py (UnicornRecorder)
#
# Edit this file to change hardware settings for the controllers and predictor.
# ============================================================================

# ====== KINOVA ARM (mirrors track_ball_kinova_test_fixed.py) ======
ROBOT_IP = "192.168.1.10"
USERNAME = "admin"
PASSWORD = "admin"
# ANGLES_INIT = [101.9, 39.18, 177.13, 246.77, 285.5, 86.98, 120.09]
ANGLES_INIT = [97.39, 41.8, 194.35, 232.9, 317.41, 86.03, 86.24]
LIMIT_M = 0.30
V_MAX = 0.2
AX_MAX = 0.80

# Camera (for Option 6 / track_ball)
CAM_INDEX = 0
CAM_URL = None
WARP_W, WARP_H = 1280, 720
PIXELS_PER_CM_BAR = 24.0
M_PER_PX_BAR = 0.01 / PIXELS_PER_CM_BAR
CALIB_SIGN = 1
HZ = 60
KP_POS = 1.2
KD_ERR = 0.20
KV_FF = 1.10
SMOOTH = 0.25
LATENCY_S = 0.060
LEAD_CLAMP_S = 0.80
VY_BOUNCE_MIN = 120.0
BOOST_MULT = 1.6
BOOST_FRAMES = 8
INVERT_X = False

# HSV thresholds (blue ball / red bar) - for Option 6
BLUE_LO = (90, 100, 100)
BLUE_HI = (100, 255, 255)
RED1_LO = (0, 120, 70)
RED1_HI = (10, 255, 255)
RED2_LO = (170, 120, 70)
RED2_HI = (180, 255, 255)

# Kalman tuning (pixels)
KF_MEAS_STD_PX = 4.0
KF_ACCEL_STD = 600.0

# Safety boundary (shared across controllers)
BOUNDARY_MARGIN_M = 0.0  # Default to exact travel limits; override if needed

# ====== EEG HEADSET (mirrors dataCollection.py UnicornRecorder) ======
EEG_FS = 250
# EEG_FS = 500
EEG_SAMPLES_PER_READ = 4
