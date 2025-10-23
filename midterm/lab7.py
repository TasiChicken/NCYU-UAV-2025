# =========================
# Drone ArUco FSM with SIMULATION (no RC sent when simulate=True)
# =========================

import cv2
import time
import enum
import math
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from djitellopy import Tello
from pyimagesearch.pid import PID

def calculate_marker_angle(rvec):
    R, _ = cv2.Rodrigues(rvec)
    marker_x_in_camera = R @ np.array([1, 0, 0])
    angle_rad_alt = math.atan2(marker_x_in_camera[1], marker_x_in_camera[0])
    return math.degrees(angle_rad_alt), R

def keyboard(drone: Tello, key: int, airborne: bool, simulation: bool):
    """
    回傳 (is_move_cmd, airborne_changed, simulation_changed)
    - 模擬模式開啟時，移動鍵只印出不送真指令
    """
    if key == -1:
        return False, None, None

    print("key:", key)
    fb_speed, lf_speed, ud_speed, degree = 40, 40, 50, 30
    is_move = False
    airborne_changed = None
    simulation_changed = None

    if key == ord('t'):
        simulation_changed = not simulation
        print(f"[KEY] toggle SIMULATION -> {simulation_changed}")
        return False, None, simulation_changed

    if key == ord('1'):
        if not simulation:
            try:
                drone.takeoff()
                time.sleep(1.0)
                airborne_changed = True
                print("[KEY] takeoff")
            except Exception as e:
                print("takeoff failed:", e)
        else:
            print("[SIM] takeoff (not actually sent)")
    elif key == ord('2'):
        if not simulation:
            try:
                drone.land()
                airborne_changed = False
                print("[KEY] land")
            except Exception as e:
                print("land failed:", e)
        else:
            print("[SIM] land (not actually sent)")
    elif key == ord('3'):
        if not simulation and airborne:
            drone.send_rc_control(0, 0, 0, 0); is_move = True
        print("stop")
    elif key == ord('w'):
        if not simulation and airborne:
            drone.send_rc_control(0, fb_speed, 0, 0); is_move = True
        print("forward")
    elif key == ord('s'):
        if not simulation and airborne:
            drone.send_rc_control(0, -fb_speed, 0, 0); is_move = True
        print("backward")
    elif key == ord('a'):
        if not simulation and airborne:
            drone.send_rc_control(-lf_speed, 0, 0, 0); is_move = True
        print("left")
    elif key == ord('d'):
        if not simulation and airborne:
            drone.send_rc_control(lf_speed, 0, 0, 0); is_move = True
        print("right")
    elif key == ord('x'):
        if not simulation and airborne:
            drone.send_rc_control(0, 0, +ud_speed, 0); is_move = True
        print("up")
    elif key == ord('z'):
        if not simulation and airborne:
            drone.send_rc_control(0, 0, -ud_speed, 0); is_move = True
        print("down")
    elif key == ord('c'):
        if not simulation and airborne:
            drone.send_rc_control(0, 0, 0, +degree); is_move = True
        print("rotate cw")
    elif key == ord('v'):
        if not simulation and airborne:
            drone.send_rc_control(0, 0, 0, -degree); is_move = True
        print("rotate ccw")

    return is_move, airborne_changed, simulation_changed

class MarkerDetector:
    def __init__(self, marker_size_cm: float, calib_path="calib_tello.xml"):
        self.marker_size = marker_size_cm
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
        self.K = fs.getNode("K").mat()
        self.D = fs.getNode("D").mat()
        fs.release()
        s = self.marker_size / 2.0
        self.objPoints = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)

    def detect(self, gray):
        corners, ids, _ = self.detector.detectMarkers(gray)
        poses = {}
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                success, rvec, tvec = cv2.solvePnP(self.objPoints, corners[i], self.K, self.D)
                if success:
                    poses[int(mid)] = (rvec, tvec)
        return ids, poses

class State(Enum):
    ASCEND_SEARCH_1      = auto()
    CENTER_ON_1          = auto()
    STRAFE_TO_FIND_2     = auto()
    CENTER_ON_2          = auto()
    FORWARD_TO_TARGET    = auto()
    STRAFE_LEFT          = auto()
    DONE                 = auto()

    FOLLOW_MARKER_ID     = auto() 
    PASS_UNDER_TABLE_3   = auto() 
    ROTATE_RIGHT_90      = auto() 
    ASCEND_LOCK_4        = auto() 
    OVERBOARD_TO_FIND_5  = auto() 

@dataclass
class Context:
    drone: Tello
    detector: MarkerDetector
    frame_read: any
    pid_lr: PID
    pid_ud: PID
    pid_fb: PID
    manual_override: bool = False
    last_ids: Optional[np.ndarray] = None
    last_poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    airborne: bool = False
    simulation: bool = True  # ★ 預設模擬開啟
    last_cmd_texts: deque = field(default_factory=lambda: deque(maxlen=1))
    params: dict = field(default_factory=lambda: {
        "ID1": 1,
        "ID2": 2,
        "ASCENT_SPEED": -20,        # 依你的慣例 ud<0 上升
        "SEARCH_RIGHT_SPEED": 20,
        "CENTER_X_TOL": 15.0,
        "CENTER_Y_TOL": 15.0,
        "TARGET_Z": 50.0,
        "Z_TOL": 8.0,
        "STRAFE_LEFT_CM": 70,
        "MAX_RC": 25

        # following
        "FOLLOW_ID": 2,             # 要跟隨的 marker
        "SEARCH_FORWARD_SPEED": 10,  # 看不到 marker 時的前進速度
        "YAW_KP": 0.6,               # 偏航角度(度) → RC yaw 速度的比例
        "YAW_TOL_DEG": 5.0,          # 視為已垂直的角度公差

        "MARKER_3": 3,              # 穿桌用的 marker
        "MARKER_4": 4,              # 牆上定位用的 marker
        "MARKER_5": 5,              # 終點判斷用的 marker
        "ROTATE_DEG": 90,           # 右轉角度
        "OVERBOARD_LR": -15,        # 往左水平移動的 rc 速度
        "OVERBOARD_UD": +8,         # 往上微升的 rc 速度
        "TRACK_STABLE_N": 5,        # 連續幀數達標才視為穩定       
    })

class DroneFSM:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.state = State.ASCEND_SEARCH_1  # ★ 直接從第一步開始（模擬/地面也跑）
        self.strafe_t0 = None
        self.done_t0 = None          # DONE 用的計時器
        self.done_phase = 0          # 0:init, 1:forwarding, 2:landing, 3:finished
        self.handlers = {
            State.ASCEND_SEARCH_1: self.handle_ASCEND_SEARCH_1,
            State.CENTER_ON_1:     self.handle_CENTER_ON_1,
            State.STRAFE_TO_FIND_2:self.handle_STRAFE_TO_FIND_2,
            State.CENTER_ON_2:     self.handle_CENTER_ON_2,
            State.FORWARD_TO_TARGET:self.handle_FORWARD_TO_TARGET,
            State.STRAFE_LEFT:     self.handle_STRAFE_LEFT,

            State.FOLLOW_MARKER_ID:       self.handle_FOLLOW_MARKER_ID,
            State.PASS_UNDER_TABLE_3:     self.handle_PASS_UNDER_TABLE_3,
            State.ROTATE_RIGHT_90:        self.handle_ROTATE_RIGHT_90,
            State.ASCEND_LOCK_4:          self.handle_ASCEND_LOCK_4,
            State.OVERBOARD_TO_FIND_5:    self.handle_OVERBOARD_TO_FIND_5,

            State.DONE:            self.handle_DONE,
        }
        self.blocking_running = False

    def clip(self, v: float) -> int:
        m = self.ctx.params["MAX_RC"]
        return int(max(-m, min(m, v)))

    def _record_cmd(self, text: str):
        print(text)
        self.ctx.last_cmd_texts.clear()
        self.ctx.last_cmd_texts.append(text)

    def send_rc(self, lr: float, fb: float, ud: float, yaw: float):
        lr, fb, ud, yaw = map(self.clip, (lr, fb, ud, yaw))
        if self.ctx.simulation or not self.ctx.airborne:
            self._record_cmd(f"[SIM RC] lr:{lr} fb:{fb} ud:{ud} yaw:{yaw}")
            return
        self.ctx.drone.send_rc_control(lr, fb, ud, yaw)

    def hover(self):
        if self.ctx.simulation or not self.ctx.airborne:
            self._record_cmd("[SIM RC] hover 0,0,0,0")
            return
        self.ctx.drone.send_rc_control(0, 0, 0, 0)

    def move_left_blocking(self, cm: int):
        if self.ctx.simulation or not self.ctx.airborne:
            self._record_cmd(f"[SIM MOVE] move_left {cm}cm")
            return
        self.ctx.drone.move_left(cm)

    def reset_pids(self):
        self.ctx.pid_lr.initialize()
        self.ctx.pid_ud.initialize()
        self.ctx.pid_fb.initialize()

    def _marker_yaw_error_deg_from_rvec(self, rvec):
        """
        Calculate yaw angle error between drone and marker.
        Returns angle in degrees: positive = need to rotate CW, negative = CCW
        """
        angle_deg, _ = calculate_marker_angle(rvec)
        
        # Normalize to [-180, 180]
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg < -180:
            angle_deg += 360
        
        # The angle represents how much the marker is rotated in the image
        # We want to align perpendicular to it, so this is our error
        return angle_deg

    def run(self):
        print(f"[FSM] Start at {self.state.name} | SIMULATION={self.ctx.simulation}")
        while True:
            frame = self.ctx.frame_read.frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            key = cv2.waitKey(1)
            if key == 27:
                break
            if key != -1:
                is_move, airborne_changed, simulation_changed = keyboard(
                    self.ctx.drone, key, self.ctx.airborne, self.ctx.simulation
                )
                self.ctx.manual_override = bool(is_move)
                if airborne_changed is not None:
                    self.ctx.airborne = airborne_changed
                if simulation_changed is not None:
                    self.ctx.simulation = simulation_changed

            ids, poses = self.ctx.detector.detect(gray)
            self.ctx.last_ids = ids
            self.ctx.last_poses = poses

            # HUD
            y0 = 25
            if self.ctx.simulation:
                cv2.putText(frame, "SIMULATION (no RC sent)", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                y0 += 25
            else:
                cv2.putText(frame, f"AIRBORNE={self.ctx.airborne}", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y0 += 25

            if ids is None:
                cv2.putText(frame, "NO MARKER DETECTED", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "MARKER DETECTED", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 最近指令回饋
            if self.ctx.last_cmd_texts:
                cv2.putText(frame, self.ctx.last_cmd_texts[-1], (10, y0 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # FSM
            if not self.ctx.manual_override:
                handler = self.handlers.get(self.state)
                if handler:
                    next_state = handler(self.ctx)
                    if next_state != self.state:
                        print(f"[FSM] {self.state.name} -> {next_state.name}")
                        self.state = next_state

            cv2.imshow("drone", frame)

        cv2.destroyAllWindows()

    # ------- handlers -------
    def handle_ASCEND_SEARCH_1(self, ctx: Context) -> State:
        frame = ctx.frame_read.frame
        cv2.putText(frame, f"STATE: ASCEND_SEARCH_1 (find ID={ctx.params['ID1']})",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if ctx.last_ids is not None and ctx.params["ID1"] in ctx.last_poses:
            self.reset_pids()
            self.hover()
            return State.CENTER_ON_1

        # 模擬或真飛都會「計算並顯示」要送的上升指令
        self.send_rc(0, 0, -ctx.params["ASCENT_SPEED"], 0)
        return State.ASCEND_SEARCH_1

    def handle_CENTER_ON_1(self, ctx: Context) -> State:
        frame = ctx.frame_read.frame
        tid = ctx.params["ID1"]
        cv2.putText(frame, f"STATE: CENTER_ON_1 (ID={tid})", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if ctx.last_ids is None or tid not in ctx.last_poses:
            self.hover()
            return State.ASCEND_SEARCH_1

        rvec, tvec = ctx.last_poses[tid]
        x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]
        lr_out = ctx.pid_lr.update(x, sleep=0.0)
        ud_out = ctx.pid_ud.update(y, sleep=0.0)
        self.send_rc(lr_out, 0, -ud_out, 0)

        if abs(x) <= ctx.params["CENTER_X_TOL"] and abs(y) <= ctx.params["CENTER_Y_TOL"]:
            self.reset_pids()
            self.hover()
            return State.STRAFE_TO_FIND_2
        return State.CENTER_ON_1

    def handle_STRAFE_TO_FIND_2(self, ctx: Context) -> State:
        frame = ctx.frame_read.frame
        cv2.putText(frame, "STATE: STRAFE_TO_FIND_2 (right to find ID=2)", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if ctx.last_ids is not None and ctx.params["ID2"] in ctx.last_poses:
            self.hover()
            self.reset_pids()
            return State.CENTER_ON_2

        self.send_rc(ctx.params["SEARCH_RIGHT_SPEED"], 0, 0, 0)
        return State.STRAFE_TO_FIND_2

    def handle_CENTER_ON_2(self, ctx: Context) -> State:
        frame = ctx.frame_read.frame
        tid = ctx.params["ID2"]
        cv2.putText(frame, f"STATE: CENTER_ON_2 (ID={tid})", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if ctx.last_ids is None or tid not in ctx.last_poses:
            self.hover()
            return State.STRAFE_TO_FIND_2

        rvec, tvec = ctx.last_poses[tid]
        x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]
        lr_out = ctx.pid_lr.update(x, sleep=0.0)
        ud_out = ctx.pid_ud.update(y, sleep=0.0)
        self.send_rc(lr_out, 0, -ud_out, 0)

        if abs(x) <= ctx.params["CENTER_X_TOL"] and abs(y) <= ctx.params["CENTER_Y_TOL"]:
            self.reset_pids()
            return State.FORWARD_TO_TARGET
        return State.CENTER_ON_2

    def handle_FORWARD_TO_TARGET(self, ctx: Context) -> State:
        frame = ctx.frame_read.frame
        tid = ctx.params["ID2"]
        cv2.putText(frame, "STATE: FORWARD_TO_TARGET", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if ctx.last_ids is None or tid not in ctx.last_poses:
            self.hover()
            return State.STRAFE_LEFT

        rvec, tvec = ctx.last_poses[tid]
        x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]
        lr_out = ctx.pid_lr.update(x, sleep=0.0)
        ud_out = ctx.pid_ud.update(y, sleep=0.0)
        fb_out = ctx.pid_fb.update(z - ctx.params["TARGET_Z"], sleep=0.0)
        self.send_rc(lr_out, fb_out, -ud_out, 0)

        if abs(z - ctx.params["TARGET_Z"]) <= ctx.params["Z_TOL"]:
            self.hover()
            return State.STRAFE_LEFT
        return State.FORWARD_TO_TARGET

    def handle_STRAFE_LEFT(self, ctx: Context) -> State:
        frame = ctx.frame_read.frame
        cv2.putText(frame,
                    f"STATE: STRAFE_LEFT (rc left ~{ctx.params['STRAFE_LEFT_CM']}cm)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 速度與時間
        v = int(max(10, 5))
        t_needed = float(3)

        # 第一次進來 → 設起始時間、清 PID、開始左移
        if self.strafe_t0 is None:
            self.reset_pids()
            self.hover()  # 先確保停住
            self.strafe_t0 = time.time()

        # 連續左移：每個循環都送一次 rc（模擬模式會只列印，不會真送）
        self.send_rc(-v, 0, 0, 0)

        # 判斷是否已達時間
        if time.time() - self.strafe_t0 >= t_needed:
            # 停止、清計時器並收尾
            self.send_rc(0, 0, 0, 0)
            self.strafe_t0 = None
            return State.FOLLOW_MARKER_ID

        # 尚未到時間，維持 STRAFE_LEFT
        return State.STRAFE_LEFT
    
    def handle_FOLLOW_MARKER_ID(self, ctx: Context) -> State:
        """
        跟隨指定 ID，並在看見 marker 時同時做中心對齊 (x,y)、距離對齊 (z)，
        並以 yaw 校正讓機頭轉到與 marker 垂直。
        """
        tid = ctx.params["FOLLOW_ID"]

        if not hasattr(self, "_follow_stable"):
            self._follow_stable = 0

        poses = getattr(ctx, "last_poses", {}) or {}
        if tid not in poses:
            # 看不到 marker：改為「緩慢前進」
            self._follow_stable = 0
            fwd_speed = int(ctx.params.get("SEARCH_FORWARD_SPEED", 10))
            self.send_rc(0, fwd_speed, 0, 0)
            return State.FOLLOW_MARKER_ID

        rvec, tvec = poses[tid]
        x = float(tvec[0][0])
        y = float(tvec[1][0])
        z = float(tvec[2][0])

        err_x = x
        err_y = y
        err_z = z - ctx.params["TARGET_Z"]

        lr = int(self.pid_lr.update(err_x, sleep=0.0))
        ud = int(self.pid_ud.update(err_y, sleep=0.0))
        fb = int(self.pid_fb.update(err_z, sleep=0.0))

        # === [CHANGED] 新增：由 rvec 推回偏航角誤差 → 轉成 yaw 速度 ===
        yaw_err_deg = self._marker_yaw_error_deg_from_rvec(rvec)
        yaw_kp = float(ctx.params.get("YAW_KP", 0.6))
        yaw_cmd = int(yaw_kp * yaw_err_deg)

        # 速度限幅
        cap = int(ctx.params["MAX_RC"])
        lr  = max(-cap, min(cap, lr))
        ud  = max(-cap, min(cap, ud))
        fb  = max(-cap, min(cap, fb))
        yaw_cmd = max(-cap, min(cap, yaw_cmd))  # === [CHANGED] ===

        # 實際送 RC；注意相機 y 與 RC z 的號誌
        self.send_rc(lr, fb, -ud, yaw_cmd)  # === [CHANGED] ===

        # 穩定達標檢查（多加一個 yaw 公差門檻）
        yaw_tol = float(ctx.params.get("YAW_TOL_DEG", 5.0))  # === [CHANGED] ===
        if (abs(err_x) <= ctx.params["CENTER_X_TOL"] and
            abs(err_y) <= ctx.params["CENTER_Y_TOL"] and
            abs(err_z) <= ctx.params["Z_TOL"] and
            abs(yaw_err_deg) <= yaw_tol):  # === [CHANGED] ===
            self._follow_stable += 1
        else:
            self._follow_stable = 0

        if self._follow_stable >= ctx.params["TRACK_STABLE_N"]:
            self.hover()
            self._follow_stable = 0
            return State.PASS_UNDER_TABLE_3

        return State.FOLLOW_MARKER_ID


    def handle_PASS_UNDER_TABLE_3(self, ctx: Context) -> State:
        """
        Pass under table:
        1. Search for MARKER_3
        2. Once detected, descend 50cm
        3. Move forward 2m
        4. Transition to ROTATE_RIGHT_90
        """
        frame = ctx.frame_read.frame
        tid = ctx.params["MARKER_3"]
        
        # Initialize phase tracking
        if not hasattr(self, "_pass_phase"):
            self._pass_phase = 0  # 0=searching, 1=descending, 2=moving_forward, 3=complete
            self._pass_t0 = None
            self._pass_initial_height = None
        
        cv2.putText(frame, f"STATE: PASS_UNDER_TABLE_3 (Phase {self._pass_phase})", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Phase 0: Search for MARKER_3
        if self._pass_phase == 0:
            poses = getattr(ctx, "last_poses", {}) or {}
            if tid not in poses:
                # Keep searching - gentle forward movement
                self.send_rc(0, 5, 0, 0)  # Slow forward
                cv2.putText(frame, f"Searching for Marker {tid}...", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                return State.PASS_UNDER_TABLE_3
            
            # Marker detected! Move to descending phase
            print(f"[PASS_TABLE] Marker {tid} detected, starting descent")
            self.hover()
            time.sleep(0.3)  # Brief pause
            self._pass_t0 = time.time()
            self._pass_phase = 1
            return State.PASS_UNDER_TABLE_3
        
        # Phase 1: Descend 50cm (using time-based control)
        # Assuming descent speed of ~20 cm/s, 50cm takes ~2.5 seconds
        if self._pass_phase == 1:
            DESCENT_TIME = 2.5  # seconds for 50cm descent
            elapsed = time.time() - self._pass_t0
            
            if elapsed < DESCENT_TIME:
                # Continue descending (ud>0 = down)
                self.send_rc(0, 0, 20, 0)  # Gentle descent
                cv2.putText(frame, f"Descending... {elapsed:.1f}s / {DESCENT_TIME}s", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                return State.PASS_UNDER_TABLE_3
            
            # Descent complete, move to forward phase
            print("[PASS_TABLE] Descent complete, moving forward 2m")
            self.hover()
            time.sleep(0.3)  # Brief pause
            self._pass_t0 = time.time()
            self._pass_phase = 2
            return State.PASS_UNDER_TABLE_3
        
        # Phase 2: Move forward 2m (200cm)
        # Assuming forward speed of ~40 cm/s, 200cm takes ~5 seconds
        if self._pass_phase == 2:
            FORWARD_TIME = 5.0  # seconds for 2m forward
            elapsed = time.time() - self._pass_t0
            
            if elapsed < FORWARD_TIME:
                # Continue moving forward
                self.send_rc(0, 40, 0, 0)  # Moderate forward speed
                cv2.putText(frame, f"Moving forward... {elapsed:.1f}s / {FORWARD_TIME}s", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                return State.PASS_UNDER_TABLE_3
            
            # Forward movement complete
            print("[PASS_TABLE] Passage complete, transitioning to ROTATE_RIGHT_90")
            self.hover()
            self._pass_phase = 0  # Reset for next time
            return State.ROTATE_RIGHT_90
        
        # Fallback
        return State.PASS_UNDER_TABLE_3

    def handle_ROTATE_RIGHT_90(self, ctx: Context) -> State:
        """
        Rotate right 90 degrees:
        1. Use yaw control to rotate clockwise
        2. Time-based rotation (assuming ~90 deg/s rotation rate)
        3. Transition to ASCEND_LOCK_4
        """
        frame = ctx.frame_read.frame
        
        # Initialize rotation tracking
        if not hasattr(self, "_rotate_phase"):
            self._rotate_phase = 0  # 0=init, 1=rotating, 2=complete
            self._rotate_t0 = None
        
        cv2.putText(frame, f"STATE: ROTATE_RIGHT_90 (Phase {self._rotate_phase})", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Phase 0: Initialize
        if self._rotate_phase == 0:
            print("[ROTATE_RIGHT] Starting 90° clockwise rotation")
            self.hover()
            time.sleep(0.3)  # Brief pause before rotation
            self._rotate_t0 = time.time()
            self._rotate_phase = 1
            return State.ROTATE_RIGHT_90
        
        # Phase 1: Rotate 90 degrees
        # Using rotation speed of 30 deg/s, 90° takes ~3 seconds
        if self._rotate_phase == 1:
            ROTATE_DEG = ctx.params.get("ROTATE_DEG", 90)
            ROTATE_SPEED = 30  # Degrees per second
            ROTATE_TIME = ROTATE_DEG / ROTATE_SPEED  # ~3 seconds for 90°
            
            elapsed = time.time() - self._rotate_t0
            
            if elapsed < ROTATE_TIME:
                # Continue rotating clockwise (positive yaw)
                self.send_rc(0, 0, 0, 30)  # yaw>0 = clockwise
                cv2.putText(frame, f"Rotating CW... {elapsed:.1f}s / {ROTATE_TIME:.1f}s", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                return State.ROTATE_RIGHT_90
            
            # Rotation complete
            print("[ROTATE_RIGHT] 90° rotation complete, moving to ASCEND_LOCK_4")
            self.hover()
            self._rotate_phase = 0  # Reset for next time
            return State.ASCEND_LOCK_4
        
        # Fallback
        return State.ROTATE_RIGHT_90

    def handle_ASCEND_LOCK_4(self, ctx: Context) -> State:
        """
        Ascend while searching for MARKER_4, then lock onto it:
        1. Continuously ascend (ud<0 = up in your convention)
        2. When MARKER_4 detected, use PID to center on it
        3. Maintain centered position for TRACK_STABLE_N frames
        4. Transition to OVERBOARD_TO_FIND_5
        """
        frame = ctx.frame_read.frame
        tid = ctx.params["MARKER_4"]
        
        # Initialize tracking
        if not hasattr(self, "_ascend_stable"):
            self._ascend_stable = 0
        
        cv2.putText(frame, f"STATE: ASCEND_LOCK_4 (find & lock ID={tid})", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        poses = getattr(ctx, "last_poses", {}) or {}
        
        if tid not in poses:
            # Marker not detected - continue ascending while searching
            self._ascend_stable = 0
            ASCEND_SPEED = 15  # Gentle ascent
            self.send_rc(0, 0, -ASCEND_SPEED, 0)  # ud<0 = up
            cv2.putText(frame, f"Ascending, searching for Marker {tid}...", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            return State.ASCEND_LOCK_4
        
        # Marker detected - perform centering with PID
        rvec, tvec = poses[tid]
        x = float(tvec[0][0])
        y = float(tvec[1][0])
        z = float(tvec[2][0])
        
        # Display marker position
        cv2.putText(frame, f"Marker4: x={x:.1f} y={y:.1f} z={z:.1f}cm", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # PID control for centering
        err_x = x
        err_y = y
        
        lr = int(ctx.pid_lr.update(err_x, sleep=0.0))
        ud = int(ctx.pid_ud.update(err_y, sleep=0.0))
        
        # Speed limiting
        cap = int(ctx.params["MAX_RC"])
        lr = max(-cap, min(cap, lr))
        ud = max(-cap, min(cap, ud))
        
        # Send control command (note: ud sign is flipped for camera-to-drone coordinate)
        self.send_rc(lr, 0, -ud, 0)
        
        # Check if centered within tolerance
        if (abs(err_x) <= ctx.params["CENTER_X_TOL"] and 
            abs(err_y) <= ctx.params["CENTER_Y_TOL"]):
            self._ascend_stable += 1
            cv2.putText(frame, f"Locked! Stable: {self._ascend_stable}/{ctx.params['TRACK_STABLE_N']}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            self._ascend_stable = 0
            cv2.putText(frame, "Centering on marker...", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Check if stable for required number of frames
        if self._ascend_stable >= ctx.params["TRACK_STABLE_N"]:
            print(f"[ASCEND_LOCK_4] Locked on Marker {tid}, moving to OVERBOARD_TO_FIND_5")
            self.hover()
            self._ascend_stable = 0  # Reset for next time
            return State.OVERBOARD_TO_FIND_5
        
        return State.ASCEND_LOCK_4

    def handle_OVERBOARD_TO_FIND_5(self, ctx: Context) -> State:
        """
        Move left to find MARKER_5:
        1. Continuously move left (lr<0)
        2. When MARKER_5 detected, hover and transition to DONE
        """
        frame = ctx.frame_read.frame
        tid = ctx.params["MARKER_5"]
        
        cv2.putText(frame, f"STATE: OVERBOARD_TO_FIND_5 (searching ID={tid})", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        poses = getattr(ctx, "last_poses", {}) or {}
        
        # Check if MARKER_5 is detected
        if tid in poses:
            # Marker 5 found! Hover and transition to DONE
            print(f"[OVERBOARD] Marker {tid} detected! Transitioning to DONE")
            self.hover()
            time.sleep(0.3)  # Brief pause for stability
            return State.DONE
        
        # Marker not detected - continue moving left only
        lr_speed = ctx.params.get("OVERBOARD_LR", -15)  # Negative = left
        
        # Move left without ascending
        self.send_rc(lr_speed, 0, 0, 0)  # Left only
        
        cv2.putText(frame, f"Moving left, searching for Marker {tid}...", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, f"lr={lr_speed}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return State.OVERBOARD_TO_FIND_5


    def handle_DONE(self, ctx: Context) -> State:  # move forward and land
        frame = ctx.frame_read.frame
        cv2.putText(frame, "STATE: DONE (forward then stop & land)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 使用與 STRAFE_LEFT 相同的速度/時間 → 距離近似相同
        v = int(max(10, min(ctx.params["MAX_RC"], ctx.params.get("STRAFE_LEFT_RC", 30))))
        t_needed = float(max(0.2, ctx.params.get("STRAFE_LEFT_TIME", 2.0)))

        # 初次進入
        if self.done_phase == 0:
            self.reset_pids()
            self.hover()
            self.done_t0 = time.time()
            self.done_phase = 1

        # 階段 1：向前持續 t_needed 秒（每迴圈送 RC）
        if self.done_phase == 1:
            self.send_rc(0, v, 0, 0)  # fb = +v → 往前
            if time.time() - self.done_t0 >= t_needed:
                # 到時間 → 停下，準備降落
                self.send_rc(0, 0, 0, 0)
                self.done_t0 = time.time()
                self.done_phase = 2
            return State.DONE

        # 階段 2：執行降落（模擬就只列印，真飛才 land 一次）
        if self.done_phase == 2:
            try:
                if ctx.simulation or not ctx.airborne:
                    self._record_cmd("[SIM MOVE] land")
                else:
                    ctx.drone.land()
                    ctx.airborne = False
            except Exception as e:
                print(f"[WARN] land failed: {e}")
            finally:
                self.done_phase = 3
            return State.DONE

        # 階段 3：已完成（維持停止/懸停狀態）
        self.hover()
        return State.DONE



def main():
    drone = Tello()
    drone.connect()
    print(f"Battery: {drone.get_battery()}%")
    drone.streamon()
    time.sleep(2.0)
    frame_read = drone.get_frame_read()

    pid_lr = PID(kP=0.5, kI=0.0, kD=0.0)
    pid_ud = PID(kP=0.5, kI=0.0, kD=0.0)
    pid_fb = PID(kP=0.4, kI=0.0, kD=0.0)
    pid_lr.initialize(); pid_ud.initialize(); pid_fb.initialize()

    detector = MarkerDetector(marker_size_cm=15, calib_path="calib_tello.xml")

    ctx = Context(
        drone=drone,
        detector=detector,
        frame_read=frame_read,
        pid_lr=pid_lr,
        pid_ud=pid_ud,
        pid_fb=pid_fb,
        airborne=False,       # 地面
        simulation=True       # ★ 模擬開啟
    )
    fsm = DroneFSM(ctx)

    try:
        try: drone.send_rc_control(0, 0, 0, 0)
        except: pass
        fsm.run()
    finally:
        try: drone.send_rc_control(0, 0, 0, 0)
        except: pass
        try: drone.streamoff()
        except: pass
        try: drone.end()
        except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
