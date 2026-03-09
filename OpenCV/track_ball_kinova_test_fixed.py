
# track_ball_kinova_test_fixed.py
import cv2, time, threading, numpy as np

# ====== FilterPy (Kalman) ======
try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
except ImportError as e:
    raise SystemExit("FilterPy not found. Install with: pip install filterpy") from e

# ====== Kortex ======
from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, Session_pb2

# ====== USER SETTINGS ======
ROBOT_IP   = "192.168.1.10"     # << set this
USERNAME   = "admin"
PASSWORD   = "admin"

# Optional start pose
ANGLES_INIT = [101.9, 39.18, 177.13, 246.77, 285.5, 86.98, 120.09]

# Camera
CAM_INDEX  = 0
CAM_URL    = None
WARP_W, WARP_H = 1280, 720

# ====== Screen & Mapping ======
# Mapping measured from your working script: 24 px of bar-y == 1 cm of robot-X
PIXELS_PER_CM_BAR = 24.0
M_PER_PX_BAR = 0.01 / PIXELS_PER_CM_BAR

# Set this so that when the ball is BELOW the bar (e_px > 0), vx_cmd drives the bar DOWN on screen.
CALIB_SIGN = +1   # flip to -1 if your bar moves opposite

# ====== Control & Prediction ======
HZ            = 60                    # control loop Hz
V_MAX         = 0.17                  # m/s speed limit
AX_MAX        = 0.80                  # m/s^2 accel (slew) limit
KP_POS        = 1.2                   # P on position error (px -> desired px/s)
KD_ERR        = 0.20                  # D on error (px/s) to damp
KV_FF         = 1.10                  # feed-forward on ball vy (px/s -> desired px/s)
SMOOTH        = 0.25                  # smoothing for signals (0..1)
LATENCY_S     = 0.060                 # total sensing+actuation latency (s), tune 40–90 ms
LEAD_CLAMP_S  = 0.80                  # don't predict beyond this
VY_BOUNCE_MIN = 120.0                 # px/s to treat a direction flip as a bounce
BOOST_MULT    = 1.6                   # temporary gain boost after bounce
BOOST_FRAMES  = 8

# Workspace corridor: lock X between start and start-LIMIT_M
LIMIT_M       = 0.35
INVERT_X      = False                 # if your robot X sign is reversed

# HSV thresholds (blue ball / red bar)
BLUE_LO = np.array([90,  100,  100]);  BLUE_HI = np.array([100,255,255])
RED1_LO = np.array([0,  120, 70]);   RED1_HI = np.array([10, 255,255])
RED2_LO = np.array([170,120, 70]);   RED2_HI = np.array([180,255,255])

# Kalman tuning (pixels)
KF_MEAS_STD_PX = 4.0
KF_ACCEL_STD   = 600.0

# ====== Utils ======
def clamp(v, lo, hi): return max(lo, min(hi, v))

def biggest_contour_center(mask, min_area=200):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area: return None, None
    x,y,w,h = cv2.boundingRect(c)
    return (x + w//2, y + h//2), (x,y,w,h)

def pick_screen_corners(frame):
    pts = []
    win = "Calibration"
    help_txt = "Click TL, TR, BR, BL. 'r' reset, 's' save, ESC cancel."
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    def on_mouse(ev,x,y,flags,param):
        nonlocal pts
        if ev == cv2.EVENT_LBUTTONUP: pts.append((x,y))
        elif ev == cv2.EVENT_RBUTTONUP and pts: pts.pop()
    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = frame.copy()
        cv2.putText(disp, help_txt, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        for i,p in enumerate(pts):
            cv2.circle(disp, p, 6, (0,255,255), -1)
            cv2.putText(disp, str(i+1), (p[0]+8,p[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)
        cv2.imshow(win, disp)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('r'): pts = []
        if k == ord('s') and len(pts)==4: break
        if k == 27: return None
    src = np.float32(pts)  # TL,TR,BR,BL
    dst = np.float32([[0,0],[WARP_W-1,0],[WARP_W-1,WARP_H-1],[0,WARP_H-1]])
    H   = cv2.getPerspectiveTransform(src, dst)
    print("Homography:\n", H)
    return H

def reflect_y_with_walls(y0, vy, t, H, r=10):
    """Predict y at time t with top/bottom reflections using triangle wave."""
    A = max(H - 2*r, 1e-6)
    y_raw = y0 + vy * t
    yprime = (y_raw - r) % (2*A)
    if yprime > A:
        yprime = 2*A - yprime
    return yprime + r

# ====== Robot helpers ======
def connect_robot():
    transport = TCPTransport(); transport.connect(ROBOT_IP, 10000)
    router = RouterClient(transport, lambda e: print("Kortex error:", e))
    session = SessionManager(router)
    info = Session_pb2.CreateSessionInfo()
    info.username, info.password = USERNAME, PASSWORD
    info.session_inactivity_timeout = 60000
    info.connection_inactivity_timeout = 2000
    session.CreateSession(info)
    base = BaseClient(router)
    base_cyc = BaseCyclicClient(router)
    base.SetServoingMode(Base_pb2.ServoingModeInformation(
        servoing_mode=Base_pb2.SINGLE_LEVEL_SERVOING))
    return transport, router, session, base, base_cyc

def go_to_angles(base, base_cyc, angles, timeout=25):
    if not angles: return True
    action = Base_pb2.Action()
    for i,val in enumerate(angles):
        j = action.reach_joint_angles.joint_angles.joint_angles.add()
        j.joint_identifier = i; j.value = float(val)
    done = threading.Event()
    handle = base.OnNotificationActionTopic(
        lambda n, e=done: e.set() if n.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT] else None,
        Base_pb2.NotificationOptions()
    )
    base.ExecuteAction(action)
    ok = done.wait(timeout)
    base.Unsubscribe(handle)
    return ok

def get_x(base): return base.GetMeasuredCartesianPose().x

def send_vx(base, vx, duration=None):
    cmd = Base_pb2.TwistCommand()
    cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    cmd.duration = 0
    t = cmd.twist
    t.linear_x = float(vx)
    t.linear_y = 0.0; t.linear_z = 0.0
    t.angular_x = 0.0; t.angular_y = 0.0; t.angular_z = 0.0
    base.SendTwistCommand(cmd)

# ====== Main ======
def main():
    # Camera + calibration
    cap = cv2.VideoCapture(CAM_URL if CAM_URL else CAM_INDEX)
    if not cap.isOpened(): raise RuntimeError("Camera not opened")
    ok, frame = cap.read()
    if not ok: raise RuntimeError("No frame for calibration")
    H_persp = pick_screen_corners(frame)
    if H_persp is None:
        print("Calibration canceled."); return

    # Robot
    transport, router, session, base, base_cyc = connect_robot()

    try:
        if not go_to_angles(base, base_cyc, ANGLES_INIT):
            print("Warning: timed out going to initial angles; continuing.")

        # Lock corridor at start
        x_start = get_x(base)
        X_MIN = x_start - LIMIT_M
        X_MAX = x_start
        if X_MIN > X_MAX: X_MIN, X_MAX = X_MAX, X_MIN
        print(f"[Workspace X] start={x_start:.3f}  allowed=[{X_MIN:.3f}, {X_MAX:.3f}]")

        period = 1.0 / HZ
        kernel = np.ones((5,5), np.uint8)

        # Kalman (x,y,vx,vy) in px
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([0., 0., 0., 0.])
        kf.P = np.diag([1e3,1e3,1e3,1e3])
        kf.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        kf.R = np.diag([KF_MEAS_STD_PX**2, KF_MEAS_STD_PX**2])
        kf_initialized = False

        prev_t = time.perf_counter()
        vx_prev = 0.0
        prev_ball_y = None
        prev_bar_y  = None
        vy_ball_s   = 0.0
        vy_err_s    = 0.0
        boost_count = 0

        print("Tracking… press 'q' to quit.")
        while True:
            t_loop0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok: break

            warp = cv2.warpPerspective(frame, H_persp, (WARP_W, WARP_H))
            hsv  = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)

            blue = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
            blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
            blue = cv2.dilate(blue, kernel, 1)

            red1 = cv2.inRange(hsv, RED1_LO, RED1_HI)
            red2 = cv2.inRange(hsv, RED2_LO, RED2_HI)
            red  = cv2.morphologyEx(cv2.bitwise_or(red1, red2), cv2.MORPH_OPEN, kernel)

            (c_ball, bb_ball) = biggest_contour_center(blue, min_area=80)
            (c_bar,  bb_bar ) = biggest_contour_center(red,  min_area=160)

            # timing
            t_now = time.perf_counter()
            dt = max(1e-3, t_now - prev_t)
            prev_t = t_now

            # Kalman F/Q with dt
            kf.F = np.array([[1,0,dt,0],
                             [0,1,0,dt],
                             [0,0,1, 0],
                             [0,0,0, 1]], dtype=float)
            q2 = Q_discrete_white_noise(dim=2, dt=dt, var=KF_ACCEL_STD**2)
            kf.Q = np.array([[q2[0,0], 0,       q2[0,1], 0      ],
                             [0,       q2[0,0], 0,       q2[0,1]],
                             [q2[1,0], 0,       q2[1,1], 0      ],
                             [0,       q2[1,0], 0,       q2[1,1]]], dtype=float)
            kf.predict()

            if c_ball is not None:
                px_meas, py_meas = float(c_ball[0]), float(c_ball[1])
                if not kf_initialized:
                    kf.x = np.array([px_meas, py_meas, 0., 0.], dtype=float)
                    kf_initialized = True
                kf.update(np.array([px_meas, py_meas], dtype=float))

            kpx, kpy, kvx, kvy = kf.x.ravel()

            # ====== FULL-SCREEN INTERCEPT AT BAR-X (works for up/down, many bounces) ======
            vx_cmd = 0.0
            dir_txt = "—"
            if c_bar is not None and kf_initialized and abs(kvx) > 1e-3:
                # Measure bar y and x
                rx, ry, rw, rh = bb_bar
                bar_cy = float(ry + rh/2.0)
                bar_px = float(rx + rw/2.0)

                # time to reach bar_x (only if ball moving toward it)
                if (bar_px - kpx) * kvx > 0:
                    t_hit = (bar_px - kpx) / kvx
                    # latency compensation
                    t_eff = max(0.0, t_hit - LATENCY_S)
                    t_eff = min(t_eff, LEAD_CLAMP_S)
                    # y prediction with wall reflections for ANY number of bounces
                    y_pred = reflect_y_with_walls(kpy, kvy, t_eff, H=WARP_H, r=10)
                else:
                    # ball moving away: just aim at current filtered y
                    t_eff = 0.0
                    y_pred = kpy

                # ----- Control law in PIXELS (desired bar y-velocity) -----
                e_px = float(y_pred - bar_cy)  # + if ball below bar
                # ball vy smoothed
                vy_ball = (0.0 if prev_ball_y is None else (kpy - prev_ball_y) / dt)
                vy_ball_s = SMOOTH*vy_ball_s + (1.0 - SMOOTH)*vy_ball
                # error derivative: how fast gap changes
                vy_bar = (0.0 if prev_bar_y is None else (bar_cy - prev_bar_y) / dt)
                d_err  = vy_ball_s - vy_bar
                vy_err_s = SMOOTH*vy_err_s + (1.0 - SMOOTH)*d_err

                # bounce detection: vy sign change with sufficient magnitude
                bounced = False
                if prev_ball_y is not None:
                    if (vy_ball_s * ((prev_ball_y - kpy) / dt if dt>0 else 0.0)) < 0 and abs(vy_ball_s) > VY_BOUNCE_MIN:
                        bounced = True
                if bounced: boost_count = BOOST_FRAMES
                boost = (BOOST_MULT if boost_count > 0 else 1.0)
                if boost_count > 0: boost_count -= 1

                v_des_pxps = boost*(KV_FF*vy_ball_s + KP_POS*e_px + KD_ERR*vy_err_s)

                # ----- Map bar-y velocity -> robot X velocity -----
                vx_cmd = CALIB_SIGN * (M_PER_PX_BAR * v_des_pxps)
                vx_cmd = clamp(vx_cmd, -V_MAX, V_MAX)

                # accel limit (slew)
                dv = vx_cmd - vx_prev
                max_dv = AX_MAX * dt
                if abs(dv) > max_dv:
                    vx_cmd = vx_prev + np.sign(dv) * max_dv
                vx_prev = vx_cmd

                # corridor guard
                x_now = get_x(base)
                if (x_now <= X_MIN and vx_cmd < 0) or (x_now >= X_MAX and vx_cmd > 0):
                    vx_cmd = 0.0

                dir_txt = f"predict@x={bar_px:.0f} t_eff={t_eff:.2f}s e={e_px:.1f}px boost={boost>1}"

                # Overlays
                cv2.line(warp, (0, int(bar_cy)), (WARP_W-1, int(bar_cy)), (255,255,255), 2)
                cv2.circle(warp, (int(bar_px), int(y_pred)), 6, (0,255,255), -1)  # predicted intercept

                prev_ball_y = kpy
                prev_bar_y  = bar_cy

            else:
                prev_ball_y = None
                prev_bar_y  = None

            # Command robot
            send_vx(base, vx_cmd, duration=0)

            # HUD
            x_now = get_x(base)
            if bb_ball:
                x,y,w,h = bb_ball; cv2.rectangle(warp, (x,y), (x+w,y+h), (255,0,0), 2)
                cv2.circle(warp, (int(kpx), int(kpy)), 5, (255,255,255), -1)
            if bb_bar:
                rx,ry,rw,rh = bb_bar; cv2.rectangle(warp, (rx,ry), (rx+rw,ry+rh), (0,0,255), 2)
                cv2.putText(warp, "Bar", (rx, max(15, ry-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            cv2.putText(warp, f"x_now={x_now:.3f} vx={vx_cmd:.3f} allowed=[{X_MIN:.3f},{X_MAX:.3f}] {dir_txt}",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

            cv2.imshow("Warped screen (KF + full-screen intercept)", warp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # keep loop at ~HZ
            elapsed = time.perf_counter() - t_loop0
            sleep_left = period - elapsed
            if sleep_left > 0: time.sleep(sleep_left)

    finally:
        try: send_vx(base, 0.0, duration=0)
        except: pass
        try: session.CloseSession()
        except: pass
        try: transport.disconnect()
        except: pass
        cap.release()
        cv2.destroyAllWindows()
        print("Disconnected cleanly.")

if __name__ == "__main__":
    main()
