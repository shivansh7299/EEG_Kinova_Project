# ============================================================================
# Kinova EEG Controller - Option 5
# ============================================================================
# Uses highest-accuracy EEG model to predict real-time and move Kinova arm.
# No OpenCV - arm movement is purely driven by EEG predictions.
# Usage: python kinova_eeg_controller.py [--model EEGNet|FBMSNet|CTNet]
# ============================================================================

import sys
import time
import argparse
import threading
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Unicorn EEG
try:
    import UnicornPy
    HAS_UNICORN = True
except ImportError:
    HAS_UNICORN = False
    print("UnicornPy not found. Install for real EEG. Using mock data.")

# Kinova (same as OpenCV/track_ball_kinova_test_fixed.py)
try:
    from kortex_api.TCPTransport import TCPTransport
    from kortex_api.RouterClient import RouterClient
    from kortex_api.SessionManager import SessionManager
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.messages import Base_pb2, Session_pb2
    HAS_KORTEX = True
except ImportError:
    HAS_KORTEX = False
    print("Kortex API not found. Install for real robot. Using mock arm.")

# Real-time predictor
from real_time_eeg_predictor import RealTimeEEGPredictor, N_CHANNELS

# ====== Settings (from hardware_config.py) ======
from hardware_config import ROBOT_IP, USERNAME, PASSWORD, ANGLES_INIT, V_MAX, LIMIT_M

PREDICTION_INTERVAL_S = 1.0


def connect_robot():
    if not HAS_KORTEX:
        return None
    transport = TCPTransport()
    transport.connect(ROBOT_IP, 10000)
    router = RouterClient(transport, lambda e: print("Kortex error:", e))
    session = SessionManager(router)
    info = Session_pb2.CreateSessionInfo()
    info.username, info.password = USERNAME, PASSWORD
    info.session_inactivity_timeout = 60000
    info.connection_inactivity_timeout = 2000
    session.CreateSession(info)
    base = BaseClient(router)
    base_cyc = BaseCyclicClient(router)
    base.SetServoingMode(Base_pb2.ServoingModeInformation(servoing_mode=Base_pb2.SINGLE_LEVEL_SERVOING))
    return transport, router, session, base, base_cyc


def go_to_angles(base, base_cyc, angles, timeout=25):
    """Move arm to initial angles (same as track_ball_kinova_test_fixed.py)."""
    if not angles or not HAS_KORTEX:
        return True
    action = Base_pb2.Action()
    for i, val in enumerate(angles):
        j = action.reach_joint_angles.joint_angles.joint_angles.add()
        j.joint_identifier = i
        j.value = float(val)
    done = threading.Event()
    handle = base.OnNotificationActionTopic(
        lambda n, e=done: e.set() if n.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT] else None,
        Base_pb2.NotificationOptions()
    )
    base.ExecuteAction(action)
    ok = done.wait(timeout)
    base.Unsubscribe(handle)
    return ok


def send_vx(base, vx):
    """Send velocity in X only (up/down bar direction, same as track_ball)."""
    if base is None:
        return
    cmd = Base_pb2.TwistCommand()
    cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    cmd.duration = 0
    t = cmd.twist
    t.linear_x = float(vx)   # X axis = bar up/down
    t.linear_y = 0.0
    t.linear_z = 0.0
    t.angular_x = 0.0
    t.angular_y = 0.0
    t.angular_z = 0.0
    base.SendTwistCommand(cmd)


def get_x(base):
    if base is None:
        return 0.0
    return base.GetMeasuredCartesianPose().x


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ====== EEG Stream ======
class EEGStream:
    """Stream EEG data from Unicorn or mock."""

    def __init__(self, predictor):
        self.predictor = predictor
        self.running = False
        self.device = None
        self.num_channels = 17  # Unicorn Hybrid Black
        self.thread = None

    def connect(self):
        if not HAS_UNICORN:
            print("Mock EEG: generating random data")
            return True
        devices = UnicornPy.GetAvailableDevices(True)
        if not devices:
            raise RuntimeError("No Unicorn device found.")
        self.device = UnicornPy.Unicorn(devices[0])
        self.num_channels = self.device.GetNumberOfAcquiredChannels()
        self.device.StartAcquisition(False)
        print(f"EEG connected: {self.num_channels} channels")
        return True

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        samples_per_read = 4
        buffer_size = samples_per_read * self.num_channels * 4
        while self.running:
            try:
                if HAS_UNICORN and self.device:
                    receive_buffer = bytearray(buffer_size)
                    self.device.GetData(samples_per_read, receive_buffer, buffer_size)
                    data = np.frombuffer(receive_buffer, dtype=np.float32)
                    data = data.reshape((samples_per_read, self.num_channels))
                else:
                    data = np.random.randn(samples_per_read, 8).astype(np.float32) * 10
                self.predictor.add_samples(data[:, :8])
            except Exception as e:
                print(f"EEG read error: {e}")
            time.sleep(1.0 / 250)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if HAS_UNICORN and self.device:
            try:
                self.device.StopAcquisition()
            except Exception:
                pass
            self.device = None


# ====== Main Loop ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["EEGNet", "FBMSNet", "CTNet"], default="CTNet")
    parser.add_argument("--no-robot", action="store_true", help="Run without robot (EEG + print only)")
    args = parser.parse_args()

    print("=" * 60)
    print("Option 5: EEG → Kinova Arm (No OpenCV)")
    print("=" * 60)

    # Load predictor
    try:
        predictor = RealTimeEEGPredictor(model_name=args.model)
        print(f"Loaded {args.model} with {predictor.num_classes} classes")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Connect robot and move to initial angles (same as track_ball)
    robot = None
    if HAS_KORTEX and not args.no_robot:
        try:
            robot = connect_robot()
            transport, router, session, base, base_cyc = robot
            if not go_to_angles(base, base_cyc, ANGLES_INIT):
                print("Warning: timed out going to initial angles; continuing.")
            x_start = get_x(base)
            X_MIN = x_start - LIMIT_M
            X_MAX = x_start
            if X_MIN > X_MAX:
                X_MIN, X_MAX = X_MAX, X_MIN
            print(f"Robot connected. Arm at initial pose. X range (up/down): [{X_MIN:.3f}, {X_MAX:.3f}]")
        except Exception as e:
            print(f"Robot connection failed: {e}")
            base = None
            X_MIN, X_MAX = -0.35, 0.0
    else:
        base = None
        X_MIN, X_MAX = -0.35, 0.0

    # Map class to velocity in X (up/down bar direction): 0=one way, 1=other way, 2=stop
    def class_to_vx(cls):
        if predictor.num_classes == 2:
            return -V_MAX if cls == 0 else V_MAX
        else:
            if cls == 0:
                return -V_MAX
            elif cls == 1:
                return V_MAX
            else:
                return 0.0

    # Start EEG stream
    stream = EEGStream(predictor)
    stream.connect()
    stream.start()

    print("Press Ctrl+C to stop.")
    try:
        last_pred_time = time.time()
        while True:
            now = time.time()
            if now - last_pred_time >= PREDICTION_INTERVAL_S:
                pred, probs = predictor.predict()
                if pred is not None:
                    vx = class_to_vx(pred)
                    vx = clamp(vx, -V_MAX, V_MAX)
                    # Clamp to workspace (X only, up/down direction)
                    if base:
                        x_now = get_x(base)
                        if (x_now <= X_MIN and vx < 0) or (x_now >= X_MAX and vx > 0):
                            vx = 0.0
                    send_vx(base, vx)
                    print(f"Pred: {pred} | vx: {vx:.3f} | probs: {[f'{p:.2f}' for p in probs]}")
                last_pred_time = now
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop()
        send_vx(base, 0.0)
        if robot:
            try:
                robot[2].CloseSession()
            except Exception:
                pass
            try:
                robot[0].disconnect()
            except Exception:
                pass
        print("Done.")


if __name__ == "__main__":
    main()
