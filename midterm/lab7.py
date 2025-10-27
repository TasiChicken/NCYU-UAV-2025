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

import logging

# 方案 A：全域把最低等級調到 WARNING（不顯示 INFO）
logging.basicConfig(level=logging.WARNING, force=True)

def calculate_marker_angle(rvec):
    R, _ = cv2.Rodrigues(rvec)
    angle_rad = math.atan2(-R[1, 0], R[0, 0])
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg, R

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
        "CENTER_Y_TOL": 20.0,
        "TARGET_Z": 120.0,
        "Z_TOL": 10.0,
        "STRAFE_RC":25,          # 側移 RC 速度
        "CREEP_FB": 15,           # 慢速前進 RC
        "MAX_RC": 40,
        "OPPOSITE_STRAFE_SIGN": 0, # 之後決定
        "PHASE": 0,             # 掃描階段計數器


        # following
        "FOLLOW_ID": 5,             # 要跟隨的 marker
        "FOLLOW_DIS": 40.0,
        "FOLLOW_ROT_SPE": 60.0,
        "FOLLOW_X_SPE": 20.0,
        "FOLLOW_Y_SPE": 25.0,
        "FOLLOW_Z_SPE": 30.0,

        "MARKER_3": 3,              # 穿桌用的 marker
        "MARKER_4": 4,              # 牆上定位用的 marker
        "MARKER_5": 5,              # 終點判斷用的 marker
        "ROTATE_DEG": 90,           # 右轉角度
        "OVERBOARD_LR": -15,        # 往左水平移動的 rc 速度
        "OVERBOARD_UD": +8,         # 往上微升的 rc 速度
        "TRACK_STABLE_N": 5,        # 連續幀數達標才視為穩定  

        "MARKER5_TARGET_Y": 0.0,   # 期望的相機座標系 y 距離（單位同 tvec，通常是 cm）
        "Y_TOL": 2.0,              # y 距離容許誤差（cm）    
        "ANGLE_TOL": 4.0,          # 角度容許誤差（deg）

        "MARKER_5_DIS": 50,
        "MARKER_5_X_TOL": 2.0,
        "MARKER_5_Y_TOL": 2.0,
        "MARKER_5_Z_TOL": 2.0
    })

class DroneFSM:
    def __init__(self, ctx: Context):
        self.state = State.FOLLOW_MARKER_ID
        self.ctx = ctx
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
        try:
            ids = ctx.last_ids 
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
        except KeyError:
            self.send_rc(0, 0, 0, 0)
            return State.CENTER_ONE
        except IndexError:
            self.send_rc(0, 0, 0, 0)
            return State.CENTER_ONE

    
    def handle_SCAN_SECOND(self, ctx):
        
        frame = ctx.frame_read.frame
        cv2.putText(frame, "SCAN_SECOND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        try:
            ids = ctx.last_ids       
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
            if time.time() - self.strafe_t0 > 1.0: 
                ctx.params["PHASE"] += 1 
                self.strafe_t0 = time.time()
            return State.SCAN_SECOND
        except KeyError:
            self.send_rc(0, 0, 0, 0)
            return State.SCAN_SECOND
    
    def handle_DECIDE_TARGET(self, ctx):
        frame = ctx.frame_read.frame
        cv2.putText(frame, "DECIDE_TARGET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        poses = ctx.last_poses
        id1_in = ctx.params["ID1"] in poses
        id2_in = ctx.params["ID2"] in poses


        # 1) 選較遠的目標
        self.ids_candidates = [ctx.params["ID1"], ctx.params["ID2"]]
        if len(poses) < 2:
            return State.SCAN_SECOND
        
        target_id = self._pick_farther_id(poses)

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
        # if target_id not in ctx.last_poses:
        #     return State.ASCEND_SEARCH
        try:
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
        except KeyError:
            self.send_rc(0, 0, 0, 0)
            return State.CENTER_ON_TARGET

    
    def handle_FORWARD_TO_TARGET(self, ctx):
        frame = ctx.frame_read.frame
        cv2.putText(frame, "FORWARD_TO_TARGET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        try:
            target_id = ctx.params["TARGET_ID"]
            rvec, tvec = ctx.last_poses[target_id]
            z = tvec[2][0]
            fb = ctx.pid_fb.update(z - 40.0, sleep=0.0)     # move to 40cm in front of marker
            self.send_rc(0, fb, 0, 0)
            #check if close enough
            if z < 45.0:
                self.reset_pids()
                return State.STRAFE_OPPOSITE
            return State.FORWARD_TO_TARGET
        except KeyError:
            self.send_rc(0, 0, 0, 0)
            return State.STRAFE_OPPOSITE
    
    def handle_STRAFE_OPPOSITE(self, ctx):
        frame = ctx.frame_read.frame
        cv2.putText(frame, "STRAFE_OPPOSITE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        try:
            if self.strafe_t0 is None:
                self.strafe_t0 = time.time()

            direction_sign = ctx.params["OPPOSITE_STRAFE_SIGN"]
            self.send_rc(direction_sign * 30, 0, 0, 0)
            if time.time() - self.strafe_t0 > 2.2:
                self.strafe_t0 = None
                return State.FOLLOW_MARKER_ID
            return State.STRAFE_OPPOSITE
        except KeyError:
            self.send_rc(0, 0, 0, 0)
            return State.STRAFE_OPPOSITE
    
    def handle_CREEP_FORWARD(self, ctx):
        frame = ctx.frame_read.frame
        cv2.putText(frame, "CREEP_FORWARD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        self.send_rc(0, ctx.params["CREEP_FB"], 0, 0)

        if ctx.last_ids is not None and ctx.params["FOLLOW_ID"] in ctx.last_poses:
            return State.FOLLOW_MARKER_ID
        
        return State.CREEP_FORWARD

    def handle_FOLLOW_MARKER_ID(self, ctx: Context) -> State:
        tid = ctx.params["FOLLOW_ID"]
        marker3 = ctx.params["MARKER_3"]
        target_dist = ctx.params["FOLLOW_DIS"]

        poses = getattr(ctx, "last_poses", {}) or {}

        if tid not in poses and marker3 in poses:
            self.hover()                  
            return State.PASS_UNDER_TABLE_3

        if tid not in poses:
            self.hover()   
            return State.FOLLOW_MARKER_ID  
        
        rvec, tvec = poses[tid]

        x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]
        
        # Calculate marker rotation angle using rvec
        angle_error, _ = calculate_marker_angle(rvec)
        angle_error = angle_error * 4.0
        
        # Calculate errors for PID tuning analysis
        error_x = x  # Left/Right error
        error_y = y  # Up/Down error
        error_z = z - target_dist  # Forward/Back error
        # print(error_z)
        
        
        # Step 2: Use PID to get control outputs
        #yaw_update = ctx.pid_lr.update(error_x, sleep=0.0)     # Left/Right movement
        #ud_update = ctx.pid_ud.update(error_y, sleep=0.0) * 1.5       # Up/Down movement
        #fb_update = ctx.pid_fb.update(error_z, sleep=0.0)        # Forward/Back movement


        # if fb_update < 0:
        #     fb_update = fb_update * 1.5
        
    

        yaw_update = error_x / 8
        ud_update = error_y / 8 
        fb_update = error_z / 8
    
        
        # Step 3: Apply speed limiting to prevent loss of control (建議限制最高速度防止失控)
        rot_speed = ctx.params["FOLLOW_ROT_SPE"]
        x_speed = ctx.params["FOLLOW_X_SPE"]
        y_speed = ctx.params["FOLLOW_Y_SPE"]
        z_speed = ctx.params["FOLLOW_Z_SPE"]
                
        # Limit yaw (left/right) speed
        if yaw_update > x_speed:
            yaw_update = x_speed
        elif yaw_update < -x_speed:
            yaw_update = -x_speed
        
        # Limit up/down speed  
        if ud_update > y_speed:
            ud_update = y_speed
        elif ud_update < -y_speed:
            ud_update = -y_speed
        
        # Limit forward/back speed
        if fb_update > z_speed:
            fb_update = z_speed
        elif fb_update < -z_speed:
            fb_update = -z_speed
        
        # Add rotation control based on marker orientation with proper angle normalization
        # Target alignment angle - we want the marker to appear level in the image
        # 0° = horizontal alignment (marker appears level)
        # Calculate shortest angle error  

                
        print(f"error x:{error_x}, error y:{error_y}, error z:{error_z}, error a:{angle_error}")
        print(f'fb err{yaw_update}')
        print(f'fb err{fb_update}')
        
        # Dead zone - don't rotate if error is small (prevents jittering)
        angle_dead_zone = 1.0  # degrees - larger dead zone for stability
        if abs(angle_error) < angle_dead_zone:
            angle_error = 0
        
        # Apply speed limiting for rotation
        if angle_error > rot_speed:
            angle_error = rot_speed
        elif angle_error < -rot_speed:
            angle_error = -rot_speed
        
        self.send_rc(int(yaw_update), int(fb_update), int(-ud_update), int(-angle_error))
        return State.FOLLOW_MARKER_ID


    def handle_PASS_UNDER_TABLE_3(self, ctx: Context) -> State:
        """
        Pass under table:
        1. Search for MARKER_3
        2. Center on MARKER_3 (x, y, and angle)
        3. Descend 50cm
        4. Move forward 2m
        5. Transition to ROTATE_RIGHT_90
        """
        frame = ctx.frame_read.frame
        tid = ctx.params["MARKER_3"]
        
        # Initialize phase tracking
        if not hasattr(self, "_pass_phase"):
            # 0=searching, 1=center&align, 2=descending, 3=moving_forward
            self._pass_phase = 0
            self._pass_t0 = None
            self._pass_initial_height = None
            self._pass_stable = 0
        
        cv2.putText(frame, f"STATE: PASS_UNDER_TABLE_3 (Phase {self._pass_phase})", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Phase 0: Search for MARKER_3
        if self._pass_phase == 0:
            poses = getattr(ctx, "last_poses", {}) or {}
            if tid not in poses:
                # Keep searching - gentle forward movement
                search_ud = ctx.params.get("PASS_SEARCH_UD", 5)
                self.send_rc(0, 0, search_ud, 0)
                cv2.putText(frame, f"Searching for Marker {tid}...", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                return State.PASS_UNDER_TABLE_3
            
            # Marker detected! Move to centering/alignment phase
            print(f"[PASS_TABLE] Marker {tid} detected, start centering & aligning")
            self.hover()
            time.sleep(0.3)  # Brief pause
            self.reset_pids()
            self._pass_stable = 0
            self._pass_phase = 1  # go to centering phase
            
            return State.PASS_UNDER_TABLE_3

        # Phase 1: Center on marker 3 (x, y, and yaw angle)
        if self._pass_phase == 1:
            poses = getattr(ctx, "last_poses", {}) or {}
            if tid not in poses:
                # Lost marker during centering: hold position and keep trying
                self.hover()
                cv2.putText(frame, f"Lost Marker {tid} during centering...", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                return State.PASS_UNDER_TABLE_3

            rvec, tvec = poses[tid]
            x = float(tvec[0][0])
            y = float(tvec[1][0])
            z = float(tvec[2][0])
            z = z - 70
            # angle alignment
            marker_angle, _ = calculate_marker_angle(rvec)
            # normalize to [-180, 180]
            angle_err = marker_angle

            # PID for x/y
            lr = int(ctx.pid_lr.update(x, sleep=0.0))
            ud = int(ctx.pid_ud.update(y, sleep=0.0))
            fb = int(ctx.pid_fb.update(z, sleep=0.0))

            # clamp
            cap = int(ctx.params.get("MAX_RC", 40))
            fb = max(-cap, min(cap, fb))
            lr = max(-cap, min(cap, lr))
            ud = max(-cap, min(cap, ud))
            yaw_cmd = angle_err  # negative to reduce error

            # keep distance constant here (fb=0)
            self.send_rc(lr*1.5, fb*1.5, -ud*1.5, yaw_cmd*1.5)

            # check tolerances (x/y and angle)
            tol_x = 5
            tol_y = 5
            tol_a = 4
            if abs(x) <= tol_x and abs(y) <= tol_y and abs(angle_err) <= tol_a:
                self._pass_stable += 1
                cv2.putText(frame, f"Centered+Aligned stable {self._pass_stable}/{ctx.params['TRACK_STABLE_N']}",
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            else:
                self._pass_stable = 0
                cv2.putText(frame, "Centering & aligning marker 3...",
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame, f"x={x:.1f} (tol {tol_x}), y={y:.1f} (tol {tol_y}), angle={angle_err:.1f} (tol {tol_a})",
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 2)

            if self._pass_stable >= int(ctx.params.get("TRACK_STABLE_N", 5)):
                print("[PASS_TABLE] Centered & aligned. Start descent")
                self.hover()
                time.sleep(0.2)
                self.reset_pids()
                self._pass_t0 = time.time()
                self._pass_phase = 2
            return State.PASS_UNDER_TABLE_3

        # Phase 2: Descend 50cm (using time-based control)
        if self._pass_phase == 2:
            DESCENT_TIME = ctx.params.get("PASS_DESCENT_TIME", 2)
            descent_ud = ctx.params.get("PASS_DESCENT_UD", -40)
            elapsed = time.time() - self._pass_t0
            
            if elapsed < DESCENT_TIME:
                # Continue descending (ud>0 = down)
                self.send_rc(0, 0, descent_ud, 0)
                cv2.putText(frame, f"Descending... {elapsed:.1f}s / {DESCENT_TIME}s", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                return State.PASS_UNDER_TABLE_3
            
            # Descent complete, move to forward phase
            print("[PASS_TABLE] Descent complete, moving forward 2m")
            self.hover()
            time.sleep(0.3)  # Brief pause
            self._pass_t0 = time.time()
            self._pass_phase = 3
            return State.PASS_UNDER_TABLE_3
        
        # Phase 3: Move forward 2m (200cm)
        if self._pass_phase == 3:
            FORWARD_TIME = ctx.params.get("PASS_FORWARD_TIME", 5.0)
            forward_fb = ctx.params.get("PASS_FORWARD_FB", 40)
            elapsed = time.time() - self._pass_t0
            
            if elapsed < FORWARD_TIME:
                # Continue moving forward
                self.send_rc(0, forward_fb, 0, 0)
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
        if self._rotate_phase == 1:
            ROTATE_DEG = ctx.params.get("ROTATE_DEG", 90)
            ROTATE_SPEED = ctx.params.get("ROTATE_SPEED", 30)  # Degrees per second
            rotate_yaw = ctx.params.get("ROTATE_YAW", 40)
            ROTATE_TIME = ROTATE_DEG / ROTATE_SPEED
            
            elapsed = time.time() - self._rotate_t0
            
            if elapsed < ROTATE_TIME:
                # Continue rotating clockwise (positive yaw)
                self.send_rc(0, 0, 0, rotate_yaw)
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
            ascend_speed = ctx.params.get("ASCEND_LOCK_SPEED", 40)
            self.send_rc(0, 0, ascend_speed, 0)  # ud<0 = up
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
        After confirming MARKER_4 is visible, ascend ~80cm, then move left to find MARKER_5:
        1. Wait for MARKER_4 visibility
        2. Ascend for OVERBOARD_ASCEND_TIME seconds (ud<0 = up)
        3. Then continuously move left (lr<0)
        4. When MARKER_5 detected, hover and transition to DONE
        """
        frame = ctx.frame_read.frame
        tid5 = ctx.params["MARKER_5"]
        tid4 = ctx.params["MARKER_4"]

        cv2.putText(frame, f"STATE: OVERBOARD_TO_FIND_5 (ID4->up 80cm, then search ID5={tid5})", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        poses = getattr(ctx, "last_poses", {}) or {}

        # Initialize phase machine: 0=await/ascend, 1=left-move/search-5
        if not hasattr(self, "_over_phase"):
            self._over_phase = 0
            self._over_t0 = None

        # Phase 0: Ensure marker 4 is seen, then ascend for configured time (approx. 80cm)
        if self._over_phase == 0:
            if tid4 not in poses:
                # Wait while holding hover until MARKER_4 becomes visible
                self.hover()
                cv2.putText(frame, f"Waiting for Marker {tid4} to start ascend...", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                return State.OVERBOARD_TO_FIND_5

            ascend_ud = int(ctx.params.get("OVERBOARD_ASCEND_UD", 40))
            ascend_time = float(ctx.params.get("OVERBOARD_ASCEND_TIME", 2.0))

            if self._over_t0 is None:
                self._over_t0 = time.time()

            elapsed = time.time() - self._over_t0
            if elapsed < ascend_time:
                # Continue ascending (ud<0 = up)
                self.send_rc(0, -5, ascend_ud, 0)
                cv2.putText(frame, f"Ascending ~80cm... {elapsed:.1f}s / {ascend_time:.1f}s", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                return State.OVERBOARD_TO_FIND_5
            else:
                # Ascend complete → move to left-move phase
                print("[OVERBOARD] Ascend complete. Begin moving left to find Marker 5.")
                self.hover()
                time.sleep(0.2)
                self._over_phase = 1

        # Phase 1: Move left and search for MARKER_5
        if self._over_phase == 1:
            # Check if MARKER_5 is detected
            if tid5 in poses:
                print(f"[OVERBOARD] Marker {tid5} detected! Transitioning to DONE")
                self.hover()
                time.sleep(0.3)  # Brief pause for stability
                return State.ALIGN_Y5_FLIP_LAND

            # Marker 5 not detected - continue moving left only
            lr_speed = int(ctx.params.get("OVERBOARD_LR", -20))  # Negative = left
            self.send_rc(lr_speed, 0, 0, 0)

            cv2.putText(frame, f"Moving left, searching for Marker {tid5}...", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, f"lr={lr_speed}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            return State.OVERBOARD_TO_FIND_5

        return State.OVERBOARD_TO_FIND_5

    def handle_ALIGN_Y5_FLIP_LAND(self, ctx: Context) -> State:
        tid = ctx.params["MARKER_5"]
        dis = ctx.params["MARKER_5_DIS"]

        poses = getattr(ctx, "last_poses", {}) or {}

        if tid not in poses:
            self.send_rc(0, -10, 0, 0)  # slowly move back if lost        
            return State.ALIGN_Y5_FLIP_LAND
        
        rvec, tvec = poses[tid]

        x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]
        
        # Calculate marker rotation angle using rvec
        marker_angle, z_prime = calculate_marker_angle(rvec)
        
        # Calculate errors for PID tuning analysis
        error_x = x  # Left/Right error
        error_y = y  # Up/Down error
        error_z = z - dis  # Forward/Back error (target 80cm)
        # print(error_z)
        
        x_m = ["MARKER_5_X_TOL"]
        y_m = ["MARKER_5_Y_TOL"]
        z_m = ["MARKER_5_Z_TOL"]
        if abs(error_z) < z_m and abs(error_x) < x_m and abs(error_y) < y_m:
            self.hover()
            self.ctx.drone.land()
            return State.DONE
        
        
        # Step 2: Use PID to get control outputs
        yaw_update = ctx.pid_lr.update(error_x, sleep=0.0)    # Left/Right movement
        ud_update = ctx.pid_ud.update(error_y, sleep=0.0)       # Up/Down movement
        fb_update = ctx.pid_fb.update(error_z, sleep=0.0)        # Forward/Back movement
    
        
        # Step 3: Apply speed limiting to prevent loss of control (建議限制最高速度防止失控)
        rot_speed = ctx.params["FOLLOW_ROT_SPE"]
        x_speed = ctx.params["FOLLOW_X_SPE"]
        y_speed = ctx.params["FOLLOW_Y_SPE"]
        z_speed = ctx.params["FOLLOW_Z_SPE"]
                
        # Limit yaw (left/right) speed
        if yaw_update > x_speed:
            yaw_update = x_speed
        elif yaw_update < -x_speed:
            yaw_update = -x_speed
        
        # Limit up/down speed  
        if ud_update > y_speed:
            ud_update = y_speed
        elif ud_update < -y_speed:
            ud_update = -y_speed
        
        # Limit forward/back speed
        if fb_update > z_speed:
            fb_update = z_speed
        elif fb_update < -z_speed:
            fb_update = -z_speed
        
        # Add rotation control based on marker orientation with proper angle normalization
        # Target alignment angle - we want the marker to appear level in the image
        # 0° = horizontal alignment (marker appears level)
        target_alignment_angle = 0.0
        
        # Calculate shortest angle error  
        angle_error = (marker_angle - target_alignment_angle) * 1.5
        
        # Dead zone - don't rotate if error is small (prevents jittering)
        angle_dead_zone = 1.0  # degrees - larger dead zone for stability
        if abs(angle_error) < angle_dead_zone:
            angle_error = 0
        
        # Apply speed limiting for rotation
        if angle_error > rot_speed:
            angle_error = rot_speed
        elif angle_error < -rot_speed:
            angle_error = -rot_speed
        
        self.send_rc(int(yaw_update), int(fb_update), int(-ud_update), int(-angle_error))
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
    except Exception as ex:
        print(ex)
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
