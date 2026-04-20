# ============================================================================
# Kinova EEG Controller - Option 5
# ============================================================================
# Uses highest-accuracy EEG model to predict real-time and move Kinova arm.
# No OpenCV - arm movement is purely driven by EEG predictions.
# Usage: python kinova_eeg_controller.py [--model EEGNet|FBMSNet|CTNet] [--model-path path/to/model.pth]
# ============================================================================

import sys
import time
import argparse
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
import torch

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
from hardware_config import (
    ROBOT_IP,
    USERNAME,
    PASSWORD,
    ANGLES_INIT,
    V_MAX,
    LIMIT_M,
    EEG_SAMPLES_PER_READ,
)

PREDICTION_INTERVAL_S = 1
STATUS_INTERVAL_S = 2.0
BOUNDARY_MARGIN_M = 0.03
COMMAND_TIMEOUT_S = 1.5


class Tee:
    """Write console output to terminal and log file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


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
    """Move arm to initial angles and report completion status."""
    if not angles or not HAS_KORTEX:
        return True, "skipped"
    action = Base_pb2.Action()
    for i, val in enumerate(angles):
        j = action.reach_joint_angles.joint_angles.joint_angles.add()
        j.joint_identifier = i
        j.value = float(val)
    status = {"event": "timeout"}

    def _on_action(n, e):
        if n.action_event == Base_pb2.ACTION_END:
            status["event"] = "end"
            e.set()
        elif n.action_event == Base_pb2.ACTION_ABORT:
            status["event"] = "abort"
            e.set()

    done = threading.Event()
    handle = base.OnNotificationActionTopic(
        lambda n, e=done: _on_action(n, e),
        Base_pb2.NotificationOptions()
    )
    base.ExecuteAction(action)
    done.wait(timeout)
    base.Unsubscribe(handle)
    return status["event"] == "end", status["event"]


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


def stop_robot_motion(base):
    """Best-effort stop so startup retries are not affected by prior commands."""
    if base is None:
        return
    try:
        send_vx(base, 0.0)
    except Exception:
        pass
    try:
        base.Stop()
    except Exception:
        pass


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
        self.total_samples = 0
        self.read_errors = 0
        self.last_error = None
        self.last_read_ts = None

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
        # Give the device a brief warm-up before first read.
        time.sleep(0.2)
        print(f"EEG connected: {self.num_channels} channels")
        return True

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        samples_per_read = max(4, int(EEG_SAMPLES_PER_READ))
        consecutive_read_errors = 0
        max_consecutive_errors = 40
        while self.running:
            try:
                if HAS_UNICORN and self.device:
                    buffer_size = samples_per_read * self.num_channels * 4
                    receive_buffer = bytearray(buffer_size)
                    self.device.GetData(samples_per_read, receive_buffer, buffer_size)
                    data = np.frombuffer(receive_buffer, dtype=np.float32)
                    data = data.reshape((samples_per_read, self.num_channels))
                    consecutive_read_errors = 0
                else:
                    data = np.random.randn(samples_per_read, 8).astype(np.float32) * 10
                self.predictor.add_samples(data[:, :8])
                self.total_samples += int(data.shape[0])
                self.last_read_ts = time.time()
            except Exception as e:
                self.read_errors += 1
                self.last_error = str(e)
                print(f"EEG read error: {e}")

                # Recover after repeated transient read errors (including buffer overflow).
                consecutive_read_errors += 1
                if HAS_UNICORN and self.device and consecutive_read_errors >= max_consecutive_errors:
                    print("Too many consecutive EEG read errors. Restarting acquisition...", flush=True)
                    try:
                        self.device.StopAcquisition()
                    except Exception:
                        pass
                    try:
                        self.device.StartAcquisition(False)
                        time.sleep(0.2)
                        consecutive_read_errors = 0
                        print("EEG acquisition restart successful.", flush=True)
                    except Exception as restart_err:
                        self.last_error = f"Restart failed: {restart_err}"
                        print(f"EEG acquisition restart failed: {restart_err}", flush=True)

                # Back off briefly only on error.
                time.sleep(0.02)

    def stats(self):
        age_s = None if self.last_read_ts is None else max(0.0, time.time() - self.last_read_ts)
        return {
            "total_samples": self.total_samples,
            "read_errors": self.read_errors,
            "last_error": self.last_error,
            "last_read_age_s": age_s,
            "buffer_len": len(self.predictor.buffer),
        }

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

    def restart_acquisition(self):
        if not (HAS_UNICORN and self.device):
            return False
        try:
            self.device.StopAcquisition()
        except Exception:
            pass
        try:
            self.device.StartAcquisition(False)
            time.sleep(0.2)
            self.last_read_ts = time.time()
            print("Watchdog: EEG acquisition restarted.", flush=True)
            return True
        except Exception as e:
            self.last_error = f"Watchdog restart failed: {e}"
            print(f"Watchdog: EEG acquisition restart failed: {e}", flush=True)
            return False


# ====== Main Loop ======
def main():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = None

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"kinova_eeg_controller_{run_stamp}.log"

    try:
        log_file = open(log_path, "a", encoding="utf-8")
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
    except Exception as e:
        # Continue without file logging if file creation fails.
        print(f"Warning: could not open log file {log_path}: {e}", file=original_stderr)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["EEGNet", "FBMSNet", "CTNet"], default="CTNet")
    parser.add_argument("--model-path", default="", help="Optional checkpoint path (.pth). Overrides default model lookup.")
    parser.add_argument("--no-robot", action="store_true", help="Run without robot (EEG + print only)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Inference device")
    args = parser.parse_args()

    print(f"Logging to: {log_path}", flush=True)

    print("=" * 60)
    print("Option 5: EEG → Kinova Arm (No OpenCV)")
    print("=" * 60)

    # Load predictor
    try:
        device = torch.device(args.device)
        predictor = RealTimeEEGPredictor(
            model_name=args.model,
            model_path=(args.model_path.strip() or None),
            device=device,
        )
        print(f"Loaded {args.model} with {predictor.num_classes} classes")
        print(f"Inference device: {args.device}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Connect robot and move to initial angles (same as track_ball)
    robot = None
    if HAS_KORTEX and not args.no_robot:
        try:
            robot = connect_robot()
            transport, router, session, base, base_cyc = robot

            # Block EEG start until the arm reaches the initial pose.
            max_init_attempts = 3
            reached_initial_pose = False
            for attempt in range(1, max_init_attempts + 1):
                print(f"Moving arm to initial pose (attempt {attempt}/{max_init_attempts})...", flush=True)
                stop_robot_motion(base)
                time.sleep(0.10)
                ok, init_event = go_to_angles(base, base_cyc, ANGLES_INIT, timeout=40)
                if ok:
                    print("Initial pose reached (ACTION_END).", flush=True)
                    reached_initial_pose = True
                    break
                print(f"Initial pose move failed (event={init_event}).", flush=True)
                stop_robot_motion(base)
                time.sleep(0.70)

            if not reached_initial_pose:
                print("Error: arm did not reach initial pose. Aborting before EEG prediction starts.", flush=True)
                try:
                    robot[2].CloseSession()
                except Exception:
                    pass
                try:
                    robot[0].disconnect()
                except Exception:
                    pass
                return

            x_start = get_x(base)
            X_MIN = x_start - LIMIT_M
            X_MAX = x_start
            if X_MIN > X_MAX:
                X_MIN, X_MAX = X_MAX, X_MIN
            X_MIN_SAFE = X_MIN + BOUNDARY_MARGIN_M
            X_MAX_SAFE = X_MAX - BOUNDARY_MARGIN_M
            if X_MIN_SAFE >= X_MAX_SAFE:
                x_mid = 0.5 * (X_MIN + X_MAX)
                X_MIN_SAFE = x_mid - 0.005
                X_MAX_SAFE = x_mid + 0.005
            print(f"Robot connected. Arm at initial pose. X range (up/down): [{X_MIN:.3f}, {X_MAX:.3f}]")
            print(f"Safety-clamped X range: [{X_MIN_SAFE:.3f}, {X_MAX_SAFE:.3f}] (margin={BOUNDARY_MARGIN_M:.3f} m)")
        except Exception as e:
            print(f"Robot connection failed: {e}")
            base = None
            X_MIN, X_MAX = -0.35, 0.0
            X_MIN_SAFE, X_MAX_SAFE = X_MIN + BOUNDARY_MARGIN_M, X_MAX - BOUNDARY_MARGIN_M
    else:
        base = None
        X_MIN, X_MAX = -0.35, 0.0
        X_MIN_SAFE, X_MAX_SAFE = X_MIN + BOUNDARY_MARGIN_M, X_MAX - BOUNDARY_MARGIN_M

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
        last_status_time = 0.0
        last_samples_seen = 0
        stagnant_intervals = 0
        last_motion_cmd_ts = time.time()
        watchdog_tripped = False
        while True:
            now = time.time()
            if base and (now - last_motion_cmd_ts) > COMMAND_TIMEOUT_S:
                if not watchdog_tripped:
                    print(
                        f"Motion watchdog: no fresh command for {COMMAND_TIMEOUT_S:.2f}s. Sending stop.",
                        flush=True,
                    )
                stop_robot_motion(base)
                watchdog_tripped = True
                last_motion_cmd_ts = now

            if now - last_status_time >= STATUS_INTERVAL_S:
                s = stream.stats()
                age_txt = "n/a" if s["last_read_age_s"] is None else f"{s['last_read_age_s']:.2f}s"
                print(
                    "EEG stream status | "
                    f"buffer={s['buffer_len']}/250 | samples={s['total_samples']} | "
                    f"last_read_age={age_txt} | read_errors={s['read_errors']}",
                    flush=True,
                )
                if s["total_samples"] == last_samples_seen:
                    stagnant_intervals += 1
                    print(
                        f"Warning: EEG sample count has not changed for {stagnant_intervals * STATUS_INTERVAL_S:.0f}s.",
                        flush=True,
                    )
                else:
                    stagnant_intervals = 0
                last_samples_seen = s["total_samples"]

                if s["last_read_age_s"] is not None and s["last_read_age_s"] > 1.0:
                    print("Watchdog: stale EEG reads detected. Attempting restart...", flush=True)
                    stream.restart_acquisition()

                if s["last_error"]:
                    print(f"Last EEG error: {s['last_error']}", flush=True)
                if base:
                    try:
                        x_now_status = get_x(base)
                        print(
                            f"Robot X status | x={x_now_status:.3f} | safe=[{X_MIN_SAFE:.3f}, {X_MAX_SAFE:.3f}]",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"Robot X read error: {e}", flush=True)
                last_status_time = now

            if now - last_pred_time >= PREDICTION_INTERVAL_S:
                print(f"Inference tick | buffer={len(predictor.buffer)}/250", flush=True)
                t0 = time.time()
                try:
                    pred, probs = predictor.predict()
                except Exception as e:
                    print(f"Prediction error: {e}", flush=True)
                    pred, probs = None, None
                dt_ms = (time.time() - t0) * 1000.0
                if dt_ms > 500:
                    print(f"Prediction latency: {dt_ms:.1f} ms", flush=True)
                if pred is not None:
                    vx = class_to_vx(pred)
                    vx = clamp(vx, -V_MAX, V_MAX)
                    blocked_by_limit = False
                    # Clamp to workspace (X only, up/down direction)
                    x_now = None
                    if base:
                        x_now = get_x(base)
                        if (x_now <= X_MIN_SAFE and vx < 0) or (x_now >= X_MAX_SAFE and vx > 0):
                            blocked_by_limit = True
                            vx = 0.0
                    send_vx(base, vx)
                    last_motion_cmd_ts = now
                    watchdog_tripped = False
                    if blocked_by_limit:
                        if x_now is not None:
                            print(
                                f"Pred: {pred} | vx: {vx:.3f} (blocked at safe X limit) | "
                                f"x={x_now:.3f} | probs: {[f'{p:.2f}' for p in probs]}",
                                flush=True,
                            )
                        else:
                            print(
                                f"Pred: {pred} | vx: {vx:.3f} (blocked at safe X limit) | "
                                f"probs: {[f'{p:.2f}' for p in probs]}",
                                flush=True,
                            )
                    else:
                        if x_now is not None:
                            print(
                                f"Pred: {pred} | vx: {vx:.3f} | x={x_now:.3f} | probs: {[f'{p:.2f}' for p in probs]}",
                                flush=True,
                            )
                        else:
                            print(
                                f"Pred: {pred} | vx: {vx:.3f} | probs: {[f'{p:.2f}' for p in probs]}",
                                flush=True,
                            )
                else:
                    send_vx(base, 0.0)
                    last_motion_cmd_ts = now
                    watchdog_tripped = False
                    print(
                        f"Waiting for enough EEG data for prediction | buffer={len(predictor.buffer)}/250",
                        flush=True,
                    )
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

        # Restore stdio and close log file cleanly.
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    main()
