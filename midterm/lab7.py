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
    ASCEND_SEARCH      = enum.auto()
    CENTER_ONE         = enum.auto()
    SCAN_SECOND        = enum.auto()
    DECIDE_TARGET      = enum.auto()
    CENTER_ON_TARGET   = enum.auto()
    FORWARD_TO_TARGET  = enum.auto()
    STRAFE_OPPOSITE    = enum.auto()
    CREEP_FORWARD      = enum.auto()

    DONE               = enum.auto()
    FOLLOW_MARKER_ID     = enum.auto() 
    PASS_UNDER_TABLE_3   = enum.auto() 
    ROTATE_RIGHT_90      = enum.auto() 
    ASCEND_LOCK_4        = enum.auto() 
    OVERBOARD_TO_FIND_5  = enum.auto() 

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
    })

class DroneFSM:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.state = State.FOLLOW_MARKER_ID  # ★ 直接從第一步開始（模擬/地面也跑）
        self.strafe_t0 = None
        self.done_t0 = None          # DONE 用的計時器
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
            # self._record_cmd(f"[SIM RC] lr:{lr} fb:{fb} ud:{ud} yaw:{yaw}")
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
        self.send_rc(0, 0, -ctx.params["ASCENT_SPEED"], 0)
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
        跟隨指定 ID，並在看見 marker 時同時做中心對齊 (x,y)、距離對齊 (z)，
        並以 yaw 校正讓機頭轉到與 marker 垂直。
        """
        tid = ctx.params["FOLLOW_ID"]

        if not hasattr(self, "_follow_stable"):
            self._follow_stable = 0

        poses = getattr(ctx, "last_poses", {}) or {}

        # === [CHANGED] 新增：提前取出 MARKER_3 id（若未在 params 內，預設 3） ===
        marker3 = int(ctx.params.get("MARKER_3", 3))  # === [CHANGED] ===

        if tid not in poses:
            # === [CHANGED] 若看不到 FOLLOW_ID，但看得到 MARKER_3，直接切換狀態 ===
            if marker3 in poses:                                # === [CHANGED] ===
                self._follow_stable = 0                         # === [CHANGED] ===
                self.hover()                                    # === [CHANGED] ===
                return State.PASS_UNDER_TABLE_3                 # === [CHANGED] ===
            # === [CHANGED] 否則維持原本「緩慢前進」行為 ===
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

        # === [CHANGED] 由 rvec 推回偏航角誤差 → 轉成 yaw 速度（保持原本邏輯） ===
        yaw_err_deg = self._marker_yaw_error_deg_from_rvec(rvec)
        yaw_kp = float(ctx.params.get("YAW_KP", 0.6))
        yaw_cmd = int(yaw_kp * yaw_err_deg)

        cap = int(ctx.params["MAX_RC"])
        lr  = max(-cap, min(cap, lr))
        ud  = max(-cap, min(cap, ud))
        fb  = max(-cap, min(cap, fb))
        yaw_cmd = max(-cap, min(cap, yaw_cmd))

        self.send_rc(lr, fb, -ud, yaw_cmd)

        yaw_tol = float(ctx.params.get("YAW_TOL_DEG", 5.0))
        if (abs(err_x) <= ctx.params["CENTER_X_TOL"] and
            abs(err_y) <= ctx.params["CENTER_Y_TOL"] and
            abs(err_z) <= ctx.params["Z_TOL"] and
            abs(yaw_err_deg) <= yaw_tol):
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
        TODO:
        - 若偵測到 ctx.params["MARKER_3"]，以 ud<0 下降；可設定安全 z 與最長時間保護。
        - 穿越完成（例如 z 小於門檻、或計時到）→ ROTATE_RIGHT_90。
        """
        return State.PASS_UNDER_TABLE_3

    def handle_ROTATE_RIGHT_90(self, ctx: Context) -> State:
        """
        TODO:
        - 以 yaw>0 送速度搭配時間估 90°，或直接呼叫 SDK rotate_clockwise(ctx.params["ROTATE_DEG"])。
        - 完成後 → ASCEND_LOCK_4。
        """
        return State.ROTATE_RIGHT_90

    def handle_ASCEND_LOCK_4(self, ctx: Context) -> State:
        """
        TODO:
        - 持續微升（ud>0），搜尋並對齊 ctx.params["MARKER_4"]。
        - 置中判準：|x|,|y| < tol 且連續 ctx.params["TRACK_STABLE_N"] 幀。
        - 達成後 → OVERBOARD_TO_FIND_5。
        """
        return State.ASCEND_LOCK_4

    def handle_OVERBOARD_TO_FIND_5(self, ctx: Context) -> State:
        """
        TODO:
        - 維持微升（ud = params["OVERBOARD_UD"]）與向左水平（lr = params["OVERBOARD_LR"]）。
        - 一旦偵測到 ctx.params["MARKER_5"]：
            * 立即 hover，並（若需要）設旗標供 DONE 使用；
            * 轉移 → DONE。
        """
        return State.OVERBOARD_TO_FIND_5


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
