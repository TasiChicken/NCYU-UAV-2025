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

class State(enum.Enum):
    ASCEND_SEARCH_1 = 0
    CENTER_ON_1 = 1
    STRAFE_TO_FIND_2 = 2
    CENTER_ON_2 = 3
    FORWARD_TO_TARGET = 4
    STRAFE_LEFT = 5
    DONE = 6

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
            return State.DONE

        # 尚未到時間，維持 STRAFE_LEFT
        return State.STRAFE_LEFT

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
