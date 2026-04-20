# ============================================================================
# Kinova EEG + OpenCV Controller - Option 6
# ============================================================================
# EEG model predicts direction. OpenCV tracks ball/bar.
# If EEG prediction MATCHES OpenCV direction → arm moves FAST.
# If EEG prediction MISMATCHES OpenCV direction → arm moves SLOW.
# Usage: python kinova_eeg_opencv_controller.py [--model EEGNet|FBMSNet|CTNet] [--model-path path/to/model.pth]
# ============================================================================

import sys
import time
import argparse
import threading
import numpy as np
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "OpenCV"))

# Unicorn
try:
    import UnicornPy
    HAS_UNICORN = True
except ImportError:
    HAS_UNICORN = False

# FilterPy (Kalman)
try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
except ImportError:
    print("Install filterpy: pip install filterpy")
    sys.exit(1)

# Kortex (same as OpenCV/track_ball_kinova_test_fixed.py)
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

from real_time_eeg_predictor import RealTimeEEGPredictor, N_CHANNELS

# ====== Settings (from hardware_config.py) ======
from hardware_config import (
    ROBOT_IP, USERNAME, PASSWORD, ANGLES_INIT,
    CAM_INDEX, CAM_URL, WARP_W, WARP_H,
    PIXELS_PER_CM_BAR, M_PER_PX_BAR, CALIB_SIGN,
    HZ, V_MAX, AX_MAX, KP_POS, KD_ERR, KV_FF, SMOOTH,
    LATENCY_S, LEAD_CLAMP_S, VY_BOUNCE_MIN, BOOST_MULT, BOOST_FRAMES,
    LIMIT_M, KF_MEAS_STD_PX, KF_ACCEL_STD,
    BLUE_LO as _BLUE_LO, BLUE_HI as _BLUE_HI,
    RED1_LO as _RED1_LO, RED1_HI as _RED1_HI,
    RED2_LO as _RED2_LO, RED2_HI as _RED2_HI,
)
BLUE_LO = np.array(_BLUE_LO)
BLUE_HI = np.array(_BLUE_HI)
RED1_LO = np.array(_RED1_LO)
RED1_HI = np.array(_RED1_HI)
RED2_LO = np.array(_RED2_LO)
RED2_HI = np.array(_RED2_HI)

V_FAST_MULT = 1.0    # When EEG matches OpenCV: full speed
V_SLOW_MULT = 0.25   # When EEG mismatches: 25% speed
PREDICTION_INTERVAL_S = 0.5  # EEG prediction every 0.5s
DIR_THRESHOLD = 0.02  # vx below this = "stop" for direction comparison


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def biggest_contour_center(mask, min_area=200):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None, None
    x, y, w, h = cv2.boundingRect(c)
    return (x + w // 2, y + h // 2), (x, y, w, h)


def reflect_y_with_walls(y0, vy, t, H, r=10):
    A = max(H - 2 * r, 1e-6)
    y_raw = y0 + vy * t
    yprime = (y_raw - r) % (2 * A)
    if yprime > A:
        yprime = 2 * A - yprime
    return yprime + r


def connect_robot():
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
    return base.GetMeasuredCartesianPose().x


def pick_screen_corners(frame):
    pts = []
    win = "Calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(ev, x, y, flags, param):
        nonlocal pts
        if ev == cv2.EVENT_LBUTTONUP:
            pts.append((x, y))
        elif ev == cv2.EVENT_RBUTTONUP and pts:
            pts.pop()

    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = frame.copy()
        cv2.putText(disp, "Click TL, TR, BR, BL. 'r' reset, 's' save, ESC cancel.", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for i, p in enumerate(pts):
            cv2.circle(disp, p, 6, (0, 255, 255), -1)
        cv2.imshow(win, disp)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('r'):
            pts = []
        if k == ord('s') and len(pts) == 4:
            break
        if k == 27:
            return None
    src = np.float32(pts)
    dst = np.float32([[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]])
    H = cv2.getPerspectiveTransform(src, dst)
    return H


def vx_to_direction(vx, num_classes=2):
    """Convert OpenCV vx (m/s) to direction class. 0=one way (up), 1=other way (down), 2=stop."""
    if num_classes == 2:
        if vx < -DIR_THRESHOLD:
            return 0
        elif vx > DIR_THRESHOLD:
            return 1
        return None  # ambiguous
    else:
        if vx < -DIR_THRESHOLD:
            return 0
        elif vx > DIR_THRESHOLD:
            return 1
        else:
            return 2


# ====== EEG Stream ======
class EEGStream:
    def __init__(self, predictor):
        self.predictor = predictor
        self.running = False
        self.device = None
        self.num_channels = 17
        self.thread = None
        self.latest_pred = None
        self.latest_probs = None

    def connect(self):
        if not HAS_UNICORN:
            return True
        devices = UnicornPy.GetAvailableDevices(True)
        if not devices:
            raise RuntimeError("No Unicorn device found.")
        self.device = UnicornPy.Unicorn(devices[0])
        self.num_channels = self.device.GetNumberOfAcquiredChannels()
        self.device.StartAcquisition(False)
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
                print(f"EEG error: {e}")
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

    def get_prediction(self):
        pred, probs = self.predictor.predict()
        if pred is not None:
            self.latest_pred = pred
            self.latest_probs = probs
        return self.latest_pred, self.latest_probs


# ====== Main ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["EEGNet", "FBMSNet", "CTNet"], default="CTNet")
    parser.add_argument("--model-path", default="", help="Optional checkpoint path (.pth). Overrides default model lookup.")
    args = parser.parse_args()

    print("=" * 60)
    print("Option 6: EEG + OpenCV (Match=Fast, Mismatch=Slow)")
    print("=" * 60)

    # Load predictor
    try:
        predictor = RealTimeEEGPredictor(
            model_name=args.model,
            model_path=(args.model_path.strip() or None),
        )
        num_classes = predictor.num_classes
        print(f"Loaded {args.model} with {num_classes} classes")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Camera
    cap = cv2.VideoCapture(CAM_URL if CAM_URL else CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened")
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("No frame")
    H_persp = pick_screen_corners(frame)
    if H_persp is None:
        print("Calibration canceled.")
        return

    # Robot (same as track_ball: initial pose, then X-only up/down movement)
    transport, router, session, base, base_cyc = connect_robot()
    if not go_to_angles(base, base_cyc, ANGLES_INIT):
        print("Warning: timed out going to initial angles; continuing.")
    x_start = get_x(base)
    X_MIN = x_start - LIMIT_M
    X_MAX = x_start
    if X_MIN > X_MAX:
        X_MIN, X_MAX = X_MAX, X_MIN
    print(f"Arm at initial pose. X range (up/down): [{X_MIN:.3f}, {X_MAX:.3f}]")

    # EEG stream
    stream = EEGStream(predictor)
    stream.connect()
    stream.start()

    period = 1.0 / HZ
    kernel = np.ones((5, 5), np.uint8)
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0., 0., 0., 0.])
    kf.P = np.diag([1e3, 1e3, 1e3, 1e3])
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    kf.R = np.diag([KF_MEAS_STD_PX ** 2, KF_MEAS_STD_PX ** 2])
    kf_initialized = False
    prev_t = time.perf_counter()
    vx_prev = 0.0
    prev_ball_y = None
    prev_bar_y = None
    vy_ball_s = 0.0
    vy_err_s = 0.0
    boost_count = 0
    last_eeg_time = time.time()

    print("Tracking... EEG match=fast, mismatch=slow. Press 'q' to quit.")
    try:
        while True:
            t_loop0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break

            warp = cv2.warpPerspective(frame, H_persp, (WARP_W, WARP_H))
            hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
            blue = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
            blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
            blue = cv2.dilate(blue, kernel, 1)
            red1 = cv2.inRange(hsv, RED1_LO, RED1_HI)
            red2 = cv2.inRange(hsv, RED2_LO, RED2_HI)
            red = cv2.morphologyEx(cv2.bitwise_or(red1, red2), cv2.MORPH_OPEN, kernel)

            (c_ball, bb_ball) = biggest_contour_center(blue, min_area=80)
            (c_bar, bb_bar) = biggest_contour_center(red, min_area=160)

            t_now = time.perf_counter()
            dt = max(1e-3, t_now - prev_t)
            prev_t = t_now

            kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
            q2 = Q_discrete_white_noise(dim=2, dt=dt, var=KF_ACCEL_STD ** 2)
            kf.Q = np.array([[q2[0, 0], 0, q2[0, 1], 0], [0, q2[0, 0], 0, q2[0, 1]],
                            [q2[1, 0], 0, q2[1, 1], 0], [0, q2[1, 0], 0, q2[1, 1]]], dtype=float)
            kf.predict()

            if c_ball is not None:
                px_meas, py_meas = float(c_ball[0]), float(c_ball[1])
                if not kf_initialized:
                    kf.x = np.array([px_meas, py_meas, 0., 0.], dtype=float)
                    kf_initialized = True
                kf.update(np.array([px_meas, py_meas], dtype=float))

            kpx, kpy, kvx, kvy = kf.x.ravel()

            vx_cmd = 0.0
            opencv_dir = None
            if c_bar is not None and kf_initialized and abs(kvx) > 1e-3:
                rx, ry, rw, rh = bb_bar
                bar_cy = float(ry + rh / 2.0)
                bar_px = float(rx + rw / 2.0)
                if (bar_px - kpx) * kvx > 0:
                    t_hit = (bar_px - kpx) / kvx
                    t_eff = max(0.0, min(t_hit - LATENCY_S, LEAD_CLAMP_S))
                    y_pred = reflect_y_with_walls(kpy, kvy, t_eff, H=WARP_H, r=10)
                else:
                    t_eff = 0.0
                    y_pred = kpy
                vy_ball = (0.0 if prev_ball_y is None else (kpy - prev_ball_y) / dt)
                vy_ball_s = SMOOTH * vy_ball_s + (1.0 - SMOOTH) * vy_ball
                vy_bar = (0.0 if prev_bar_y is None else (bar_cy - prev_bar_y) / dt)
                d_err = vy_ball_s - vy_bar
                vy_err_s = SMOOTH * vy_err_s + (1.0 - SMOOTH) * d_err
                bounced = False
                if prev_ball_y is not None and abs(vy_ball_s) > VY_BOUNCE_MIN:
                    if vy_ball_s * ((prev_ball_y - kpy) / dt if dt > 0 else 0.0) < 0:
                        bounced = True
                if bounced:
                    boost_count = BOOST_FRAMES
                boost = BOOST_MULT if boost_count > 0 else 1.0
                if boost_count > 0:
                    boost_count -= 1
                v_des_pxps = boost * (KV_FF * vy_ball_s + KP_POS * (y_pred - bar_cy) + KD_ERR * vy_err_s)
                vx_cmd = CALIB_SIGN * (M_PER_PX_BAR * v_des_pxps)
                vx_cmd = clamp(vx_cmd, -V_MAX, V_MAX)
                dv = vx_cmd - vx_prev
                max_dv = AX_MAX * dt
                if abs(dv) > max_dv:
                    vx_cmd = vx_prev + np.sign(dv) * max_dv
                vx_prev = vx_cmd
                x_now = get_x(base)
                if (x_now <= X_MIN and vx_cmd < 0) or (x_now >= X_MAX and vx_cmd > 0):
                    vx_cmd = 0.0
                opencv_dir = vx_to_direction(vx_cmd, num_classes)
                prev_ball_y = kpy
                prev_bar_y = bar_cy
            else:
                prev_ball_y = None
                prev_bar_y = None

            # EEG prediction
            eeg_pred = None
            if t_now - last_eeg_time >= PREDICTION_INTERVAL_S:
                eeg_pred, _ = stream.get_prediction()
                last_eeg_time = t_now

            # Match/mismatch: scale vx_cmd when both EEG and OpenCV have a direction
            status = ""
            if opencv_dir is not None and eeg_pred is not None:
                match = (opencv_dir == eeg_pred)
                mult = V_FAST_MULT if match else V_SLOW_MULT
                vx_cmd = vx_cmd * mult
                status = " MATCH (fast)" if match else " MISMATCH (slow)"

            send_vx(base, vx_cmd)

            # HUD
            if bb_ball:
                x, y, w, h = bb_ball
                cv2.rectangle(warp, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if bb_bar:
                rx, ry, rw, rh = bb_bar
                cv2.rectangle(warp, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
            info = f"vx={vx_cmd:.3f} EEG={eeg_pred} OpenCV={opencv_dir}"
            if opencv_dir is not None and eeg_pred is not None:
                info += f" {status}"
            cv2.putText(warp, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow("EEG+OpenCV Kinova", warp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed = time.perf_counter() - t_loop0
            if period - elapsed > 0:
                time.sleep(period - elapsed)

    finally:
        stream.stop()
        send_vx(base, 0.0)
        session.CloseSession()
        transport.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
