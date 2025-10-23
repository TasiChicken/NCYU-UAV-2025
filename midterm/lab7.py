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

    # print("key:", key)
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

class State(enum.Enum):
    ASCEND_SEARCH        = enum.auto()
    CENTER_ONE           = enum.auto()
    SCAN_SECOND          = enum.auto()
    DECIDE_TARGET        = enum.auto()
    CENTER_ON_TARGET     = enum.auto()
    FORWARD_TO_TARGET    = enum.auto()
    STRAFE_OPPOSITE      = enum.auto()
    CREEP_FORWARD        = enum.auto()
    DONE                 = enum.auto()

    FOLLOW_MARKER_ID     = enum.auto() 
    PASS_UNDER_TABLE_3   = enum.auto() 
    ROTATE_RIGHT_90      = enum.auto() 
    ASCEND_LOCK_4        = enum.auto() 
    OVERBOARD_TO_FIND_5  = enum.auto() 

    ALIGN_Y5_FLIP_LAND   = enum.auto()

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
        "TARGET_ID": None,
        "ASCENT_SPEED": 40,        
        "SEARCH_RIGHT_SPEED": 30,
        "CENTER_X_TOL": 15.0,
        "CENTER_Y_TOL": 15.0,
        "TARGET_Z": 70.0,
        "Z_TOL": 8.0,
        "STRAFE_LEFT_CM": 70,
        "STRAFE_RC": 35,          # 側移 RC 速度
        "STRAFE_TIME_1M": 2.2,    # 估 2.2 秒 ≈ 1m（需實機校正）
        "STRAFE_TIME_SCAN": 1.0,   # 側移掃描時間
        "CREEP_FB": 12,           # 慢速前進 RC
        "MAX_RC": 40,
        "OPPOSITE_STRAFE_SIGN": 0, # 之後決定
        "PHASE": 0,             # 掃描階段計數器


        # following
        "FOLLOW_ID": 1,             # 要跟隨的 marker
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

        "MARKER5_TARGET_Y": 0.0,   # 期望的相機座標系 y 距離（單位同 tvec，通常是 cm）
        "Y_TOL": 5.0,              # y 距離容許誤差（cm）     
    })

class DroneFSM:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.state = State.FOLLOW_MARKER_ID  # ★ 直接從第一步開始（模擬/地面也跑）
        self.strafe_t0 = None
        self.handlers = {
            
            State.ASCEND_SEARCH      : self.handle_ASCEND_SEARCH,
            State.CENTER_ONE         : self.handle_CENTER_ONE,
            State.SCAN_SECOND        : self.handle_SCAN_SECOND,
            State.DECIDE_TARGET      : self.handle_DECIDE_TARGET,
            State.CENTER_ON_TARGET   : self.handle_CENTER_ON_TARGET,
            State.FORWARD_TO_TARGET  : self.handle_FORWARD_TO_TARGET,
            State.STRAFE_OPPOSITE    : self.handle_STRAFE_OPPOSITE,
            State.CREEP_FORWARD      : self.handle_CREEP_FORWARD,

            State.FOLLOW_MARKER_ID:       self.handle_FOLLOW_MARKER_ID,
            State.PASS_UNDER_TABLE_3:     self.handle_PASS_UNDER_TABLE_3,
            State.ROTATE_RIGHT_90:        self.handle_ROTATE_RIGHT_90,
            State.ASCEND_LOCK_4:          self.handle_ASCEND_LOCK_4,
            State.OVERBOARD_TO_FIND_5:    self.handle_OVERBOARD_TO_FIND_5,

            State.ALIGN_Y5_FLIP_LAND:     self.handle_ALIGN_Y5_FLIP_LAND,

            State.DONE:            self.handle_DONE,
        }
        self.blocking_running = False

    def clip(self, v: float) -> int:
        m = self.ctx.params["MAX_RC"]
        return int(max(-m, min(m, v)))

    def _record_cmd(self, text: str):
        # print(text)
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

    def _pick_farther_id(self, poses: Dict[int, Tuple[np.ndarray,np.ndarray]]) -> Optional[int]:
        # 用 z（越大越遠）；也可改 np.linalg.norm(tvec)
        best_id, best_z = None, -1
        for mid, (_, tvec) in poses.items():
            if mid in self.ids_candidates:
                z = float(tvec[2][0])
                if z > best_z:
                    best_z, best_id = z, mid
        return best_id
    
    # Handlers for each state
    def handle_ASCEND_SEARCH(self, ctx):
        frame = ctx.frame_read.frame
        cv2.putText(frame, "ASCEND_SEARCH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        if ctx.last_ids is not None and (ctx.params["ID1"] in ctx.last_poses or ctx.params["ID2"] in ctx.last_poses):
            return State.CENTER_ONE

        # 依你的約定 ud<0 上升
        self.send_rc(0, 0, ctx.params["ASCENT_SPEED"], 0)
        return State.ASCEND_SEARCH
    def handle_CENTER_ONE(self, ctx):
        frame = ctx.frame_read.frame
        cv2.putText(frame, "CENTER_ONE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        ids = ctx.last_ids
        if ids is None:
            return State.ASCEND_SEARCH
        
        #if seen both
        if ctx.params["ID1"] in ctx.last_poses and ctx.params["ID2"] in ctx.last_poses:
            return State.DECIDE_TARGET
        #if seen one, center on it
        vis_ids = [i for i in [ctx.params["ID1"], ctx.params["ID2"]] if i in ctx.last_poses]
        tid = vis_ids[0]
        rvec, tvec = ctx.last_poses[tid]
        x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]
        lr = ctx.pid_lr.update(x, 0.0)
        ud = ctx.pid_ud.update(y, 0.0)
        self.send_rc(lr, 0, -ud, 0)
        #check if centered
        if abs(x) < ctx.params["CENTER_X_TOL"] and abs(y) < ctx.params["CENTER_Y_TOL"]:
            return State.SCAN_SECOND
        return State.CENTER_ONE
    
    def handle_SCAN_SECOND(self, ctx):
        frame = ctx.frame_read.frame
        cv2.putText(frame, "SCAN_SECOND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        ids = ctx.last_ids
        if ids is None:
            return State.ASCEND_SEARCH
        
        #if seen both
        if ctx.params["ID1"] in ctx.last_poses and ctx.params["ID2"] in ctx.last_poses:
            self.strafe_t0 = None #reset strafe timer
            return State.DECIDE_TARGET
        
        # go left and go right to scan
        v = ctx.params["STRAFE_RC"]
        phase = ctx.params["PHASE"]
        sign = 1 if phase % 2 == 0 else -1
        self.send_rc(sign * v, 0, 0, 0)  
        if self.strafe_t0 is None:
            self.strafe_t0 = time.time()
        if time.time() - self.strafe_t0 > ctx.params["STRAFE_TIME_SCAN"]:
            ctx.params["PHASE"] += 1 
            self.strafe_t0 = time.time()
        return State.SCAN_SECOND
    
    def handle_DECIDE_TARGET(self, ctx):
        frame = ctx.frame_read.frame
        cv2.putText(frame, "DECIDE_TARGET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        poses = ctx.last_poses
        id1_in = ctx.params["ID1"] in poses
        id2_in = ctx.params["ID2"] in poses
        if not (id1_in and id2_in):
            return State.ASCEND_SEARCH

        # 1) 選較遠的目標
        self.ids_candidates = [ctx.params["ID1"], ctx.params["ID2"]]
        target_id = self._pick_farther_id(poses)
        if target_id is None:
            return State.ASCEND_SEARCH

        ctx.params["TARGET_ID"] = target_id

        # 2) 依目標當下的左右位置，決定「丟失時要側移的相反方向」
        #    x>0 = 目標在右 → 丟失時往左掃（-1）；x<0 = 目標在左 → 丟失時往右掃（+1）
        x_far = float(poses[target_id][1][0][0])
        if x_far > 0:
            ctx.params["OPPOSITE_STRAFE_SIGN"] = -1   # 之後 recover/scan 用：左
        else:
            ctx.params["OPPOSITE_STRAFE_SIGN"] = +1   # 之後 recover/scan 用：右

        return State.CENTER_ON_TARGET
    
    def handle_CENTER_ON_TARGET(self, ctx):
        frame = ctx.frame_read.frame
        cv2.putText(frame, "CENTER_ON_TARGET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        target_id = ctx.params["TARGET_ID"]
        if target_id not in ctx.last_poses:
            return State.ASCEND_SEARCH

        rvec, tvec = ctx.last_poses[target_id]
        x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]
        lr = ctx.pid_lr.update(x, 0.0)
        ud = ctx.pid_ud.update(y, 0.0)
        fb = ctx.pid_fb.update(z - ctx.params["TARGET_Z"], sleep=0.0)
        self.send_rc(lr, fb, -ud, 0)
        #check if centered
        if abs(x) < ctx.params["CENTER_X_TOL"] and abs(y) < ctx.params["CENTER_Y_TOL"] and abs(z - ctx.params["TARGET_Z"]) < ctx.params["Z_TOL"]:
            self.reset_pids()
            return State.FORWARD_TO_TARGET
        return State.CENTER_ON_TARGET
    
    def handle_FORWARD_TO_TARGET(self, ctx):
        frame = ctx.frame_read.frame
        cv2.putText(frame, "FORWARD_TO_TARGET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        target_id = ctx.params["TARGET_ID"]
        if target_id not in ctx.last_poses: #go too far, lost target, still go opposite
            return State.STRAFE_OPPOSITE

        rvec, tvec = ctx.last_poses[target_id]
        z = tvec[2][0]
        fb = ctx.pid_fb.update(z - 20.0, sleep=0.0)     # move to 20cm in front of marker
        self.send_rc(0, fb, 0, 0)
        #check if close enough
        if z < 45.0:
            self.reset_pids()
            return State.STRAFE_OPPOSITE
        return State.FORWARD_TO_TARGET
    
    def handle_STRAFE_OPPOSITE(self, ctx):
        frame = ctx.frame_read.frame
        cv2.putText(frame, "STRAFE_OPPOSITE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        if self.strafe_t0 is None:
            self.strafe_t0 = time.time()

        direction_sign = ctx.params["OPPOSITE_STRAFE_SIGN"]
        self.send_rc(-direction_sign * 35, 0, 0, 0)
        if time.time() - self.strafe_t0 > 2.2:
            self.strafe_t0 = None
            return State.CREEP_FORWARD
        return State.STRAFE_OPPOSITE
    
    def handle_CREEP_FORWARD(self, ctx):
        frame = ctx.frame_read.frame
        cv2.putText(frame, "CREEP_FORWARD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        self.send_rc(0, ctx.params["CREEP_FB"], 0, 0)

        if ctx.last_ids is not None and ctx.params["FOLLOW_ID"] in ctx.last_poses:
            return State.FOLLOW_MARKER_ID
        
        return State.CREEP_FORWARD

    def handle_FOLLOW_MARKER_ID(self, ctx: Context) -> State:
        """
        跟隨指定 ID；若『看不到 FOLLOW_ID、但看到 MARKER_3』→ 立刻轉入 PASS_UNDER_TABLE_3。
        其他時候依 lab6 風格：x/y/z 用 PID，角度用 calculate_marker_angle 校正旋轉。
        """
        tid = ctx.params["FOLLOW_ID"]

        if not hasattr(self, "_follow_stable"):
            self._follow_stable = 0

        poses = getattr(ctx, "last_poses", {}) or {}

        # === [CHANGED] 先抓 marker3 id（預設 3）
        marker3 = int(ctx.params.get("MARKER_3", 3))  # === [CHANGED] ===

        # === [CHANGED] 分支順序改為：若「看不到 FOLLOW_ID 且 看得到 MARKER_3」→ 直接切狀態
        if tid not in poses and marker3 in poses:      # === [CHANGED] ===
            self._follow_stable = 0                    # === [CHANGED] ===
            self.hover()                               # === [CHANGED] ===
            return State.PASS_UNDER_TABLE_3            # === [CHANGED] ===

        if tid not in poses:
            # 看不到兩者：維持「緩慢前進」以便重新遇到目標
            self._follow_stable = 0
            fwd_speed = int(ctx.params.get("SEARCH_FORWARD_SPEED", 10))
            self.send_rc(0, fwd_speed, 0, 0)
            return State.FOLLOW_MARKER_ID

        # 下面依 lab6 風格做 PID + 角度校正
        rvec, tvec = poses[tid]
        x = float(tvec[0][0])
        y = float(tvec[1][0])
        z = float(tvec[2][0])

        err_x = x
        err_y = y
        err_z = z - ctx.params["TARGET_Z"]

        # 與 lab6 一致：x→lr, y→ud, z→fb
        lr = int(ctx.pid_lr.update(err_x, sleep=0.0))
        ud = int(ctx.pid_ud.update(err_y, sleep=0.0))
        fb = int(ctx.pid_fb.update(err_z, sleep=0.0))

        marker_angle, _ = calculate_marker_angle(rvec)        # === [CHANGED] ===
        target_alignment_angle = 0.0                          # === [CHANGED] ===
        angle_error = marker_angle - target_alignment_angle   # === [CHANGED] ===
        angle_dead_zone = float(ctx.params.get("YAW_TOL_DEG", 5.0))  # === [CHANGED] ===
        if abs(angle_error) < angle_dead_zone:                # === [CHANGED] ===
            angle_error = 0                                   # === [CHANGED] ===

        cap = int(ctx.params.get("MAX_RC", 25))
        lr  = max(-cap, min(cap, lr))
        ud  = max(-cap, min(cap, ud))
        fb  = max(-cap, min(cap, fb))
        angle_error = max(-cap, min(cap, angle_error))        # === [CHANGED] ===

        self.send_rc(lr, fb, -ud, int(-angle_error))          # === [CHANGED] ===

        # 保留原本「有跟上 FOLLOW_ID 時」的穩定判斷（對 x/y/z/角度）
        if (abs(err_x) <= ctx.params["CENTER_X_TOL"] and
            abs(err_y) <= ctx.params["CENTER_Y_TOL"] and
            abs(err_z) <= ctx.params["Z_TOL"] and
            abs(angle_error) == 0):
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

    # === [CHANGED] 新增：對齊到距離 marker 5 的 Y，翻滾後降落 ===
    def handle_ALIGN_Y5_FLIP_LAND(self, ctx: Context) -> State:
        frame = ctx.frame_read.frame
        cv2.putText(frame, "ALIGN_Y5_FLIP_LAND", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        m5 = int(ctx.params.get("MARKER_5", 5))
        poses = ctx.last_poses or {}

        # 初始化穩定幀計數
        if not hasattr(self, "_y5_stable"):
            self._y5_stable = 0

        # 沒看到 marker 5：保持懸停（或可依需求微調前進/上升）
        if m5 not in poses:
            self._y5_stable = 0
            self.hover()
            return State.ALIGN_Y5_FLIP_LAND

        # 讀取 y 距離並與目標比較（OpenCV 相機座標：y 往下為正）
        rvec, tvec = poses[m5]
        y_cur = float(tvec[1][0])                       # 目前相機座標系下的 y（cm）
        y_set = float(ctx.params["MARKER5_TARGET_Y"])   # 期望 y（cm）
        err_y = y_cur - y_set                           # 正值→需要向上或向下：交給 PID 決定

        # 只調整上下（ud），其餘維持 0；你也可加上 x/z 保持對齊，這裡依需求最小化
        ud = int(ctx.pid_ud.update(err_y, sleep=0.0))
        ud = self.clip(ud)
        self.send_rc(0, 0, -ud, 0)   # 注意你的慣例：RC 要送 -ud

        # 判斷是否到目標 y（用 Y_TOL，並可疊加 TRACK_STABLE_N 框定）
        y_tol = float(ctx.params.get("Y_TOL", 5.0))
        if abs(err_y) <= y_tol:
            self._y5_stable += 1
        else:
            self._y5_stable = 0

        if self._y5_stable >= int(ctx.params.get("TRACK_STABLE_N", 3)):
            # 對齊完成：停止，翻滾一次，然後降落
            self.hover()
            self._y5_stable = 0
            self.reset_pids()

            # 翻滾（模擬模式只記錄文字；真機則呼叫 flip）
            if ctx.simulation or not ctx.airborne:
                self._record_cmd("[SIM MOVE] flip forward")
            else:
                try:
                    ctx.drone.flip('f')   # 'f','b','l','r' 依需求可改
                except Exception as e:
                    print("flip failed:", e)

            # 降落
            if ctx.simulation or not ctx.airborne:
                self._record_cmd("[SIM MOVE] land")
            else:
                try:
                    ctx.drone.land()
                    ctx.airborne = False
                except Exception as e:
                    print("land failed:", e)

            return State.DONE

        return State.ALIGN_Y5_FLIP_LAND

    def handle_DONE(self, ctx: Context) -> State:  # move forward and land
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
