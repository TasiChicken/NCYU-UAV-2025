# =========================
# Drone ArUco FSM Template
# =========================

import time
import enum
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from djitellopy import Tello
import numpy as np
import cv2
import logging
import os
from pyimagesearch.pid import PID
from config import DroneConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_marker_angle(rvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    angle_rad = math.atan2(-rotation_matrix[1, 0], rotation_matrix[0, 0])
    angle_deg = math.degrees(angle_rad)

    return angle_deg, rotation_matrix


def keyboard(drone: Tello, key: int, airborne: bool, simulation: bool) -> Tuple[bool, Optional[bool], Optional[bool]]:
    """
    Handle keyboard input for drone control.
    Returns: (is_move_cmd, airborne_changed, simulation_changed)
    - When simulation is True, movement keys only print to console without sending commands.
    """
    if key == -1:
        return False, None, None

    fb_speed = DroneConfig.FB_SPEED
    lf_speed = DroneConfig.LF_SPEED
    ud_speed = DroneConfig.UD_SPEED
    degree = DroneConfig.DEGREE
    is_move = False
    airborne_changed = None
    simulation_changed = None

    if key == ord("t"):
        simulation_changed = not simulation
        print(f"[KEY] toggle SIMULATION -> {simulation_changed}")
        return False, None, simulation_changed

    if key == ord("1"):
        if not simulation:
            drone.takeoff()
            time.sleep(1.0)
            airborne_changed = True
            print("[KEY] takeoff")
        else:
            print("[SIM] takeoff (not actually sent)")
    elif key == ord("2"):
        if not simulation:
            drone.land()
            airborne_changed = False
            print("[KEY] land")
        else:
            print("[SIM] land (not actually sent)")
    elif key == ord("3"):
        if not simulation and airborne:
            drone.send_rc_control(0, 0, 0, 0)
            is_move = True
        print("stop")
    elif key == ord("w"):
        if not simulation and airborne:
            drone.send_rc_control(0, fb_speed, 0, 0)
            is_move = True
        print("forward")
    elif key == ord("s"):
        if not simulation and airborne:
            drone.send_rc_control(0, -fb_speed, 0, 0)
            is_move = True
        print("backward")
    elif key == ord("a"):
        if not simulation and airborne:
            drone.send_rc_control(-lf_speed, 0, 0, 0)
            is_move = True
        print("left")
    elif key == ord("d"):
        if not simulation and airborne:
            drone.send_rc_control(lf_speed, 0, 0, 0)
            is_move = True
        print("right")
    elif key == ord("x"):
        if not simulation and airborne:
            drone.send_rc_control(0, 0, +ud_speed, 0)
            is_move = True
        print("up")
    elif key == ord("z"):
        if not simulation and airborne:
            drone.send_rc_control(0, 0, -ud_speed, 0)
            is_move = True
        print("down")
    elif key == ord("c"):
        if not simulation and airborne:
            drone.send_rc_control(0, 0, 0, +degree)
            is_move = True
        print("rotate cw")
    elif key == ord("v"):
        if not simulation and airborne:
            drone.send_rc_control(0, 0, 0, -degree)
            is_move = True
        print("rotate ccw")
    elif key == ord("e"):  # Emergency Stop
        if not simulation:
            drone.send_rc_control(0, 0, 0, 0)
            drone.land()
            airborne_changed = False
            print("[KEY] EMERGENCY STOP triggered")
        else:
            print("[SIM] EMERGENCY STOP triggered")
        return False, airborne_changed, simulation_changed

    return is_move, airborne_changed, simulation_changed


class MarkerDetector:
    """
    Handles ArUco marker detection and pose estimation.
    """
    def __init__(self, marker_size_cm: float, calib_path: str = "calib_tello.xml"):
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")
            
        self.marker_size = marker_size_cm
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.aruco_params
        )
        file_storage = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
        self.camera_intrinsics_matrix = file_storage.getNode("K").mat()
        self.distortion_coefficients = file_storage.getNode("D").mat()
        file_storage.release()
        half_marker_size = self.marker_size / 2.0
        self.obj_points = np.array(
            [[-half_marker_size, half_marker_size, 0], [half_marker_size, half_marker_size, 0],
            [half_marker_size, -half_marker_size, 0], [-half_marker_size, -half_marker_size, 0]], dtype=np.float32
        )

    def detect(self, gray: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[int, Tuple[np.ndarray, np.ndarray]]]:
        """
        Detect markers in a grayscale image.
        Returns: (ids, poses_dict) where poses_dict maps ID -> (rvec, tvec)
        """
        corners, ids, _ = self.detector.detectMarkers(gray)
        poses = {}
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points,
                    corners[i],
                    self.camera_intrinsics_matrix,
                    self.distortion_coefficients,
                )
                if success:
                    poses[int(mid)] = (rvec, tvec)
        return ids, poses


class State(enum.Enum):
    INIT = enum.auto()
    TAKEOFF_SEARCH_1 = enum.auto()
    ALIGN_MARKER_1_INITIAL = enum.auto()
    MOVE_RIGHT_50 = enum.auto()
    FOLLOW_LINE = enum.auto()
    ALIGN_MARKER_1_FINAL = enum.auto()
    LAND = enum.auto()
    DONE = enum.auto()


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
    simulation: bool = True  # Default to simulation mode
    last_cmd_texts: deque = field(default_factory=lambda: deque(maxlen=1))
    config: DroneConfig = field(default_factory=DroneConfig)


class DroneFSM:
    def __init__(self, ctx: Context):
        self.state = State.INIT
        self.ctx = ctx
        
        self.handlers = {
            State.INIT: self.handle_INIT,
            State.TAKEOFF_SEARCH_1: self.handle_TAKEOFF_SEARCH_1,
            State.ALIGN_MARKER_1_INITIAL: self.handle_ALIGN_MARKER_1_INITIAL,
            State.MOVE_RIGHT_50: self.handle_MOVE_RIGHT_50,
            State.FOLLOW_LINE: self.handle_FOLLOW_LINE,
            State.ALIGN_MARKER_1_FINAL: self.handle_ALIGN_MARKER_1_FINAL,
            State.LAND: self.handle_LAND,
            State.DONE: self.handle_DONE,
        }
        self.blocking_running = False

        # State variables
        self._stable_count: int = 0
        self._state_t0: float = 0.0

        # Monitoring
        self._last_battery_check: float = 0.0

    def _validate_transition(self, from_state: State, to_state: State) -> bool:
        """Validate if state transition is allowed."""
        # Define allowed transitions (simplified for now, can be expanded)
        # For now, allow all transitions but log them
        return True

    def _apply_deadzone(self, value: float, threshold: float = 8.0) -> float:
        """Apply deadzone to RC control value to avoid drift."""
        if abs(value) < threshold:
            return 0.0
        if 0 < value < threshold:
            return threshold
        if -threshold < value < 0:
            return -threshold
        return value

    def _clamp_speed(self, value: float, max_speed: float) -> float:
        """Clamp speed value to [-max_speed, max_speed]."""
        return max(-max_speed, min(max_speed, value))

    def _update_pid_controls(
        self, 
        ctx: Context, 
        error_x: float, 
        error_y: float, 
        error_z: float
    ) -> Tuple[float, float, float]:
        """Update all PID controllers and return (lr, ud, fb) commands."""
        lr = ctx.pid_lr.update(error_x, sleep=0.0)
        ud = ctx.pid_ud.update(error_y, sleep=0.0)
        fb = ctx.pid_fb.update(error_z, sleep=0.0)
        return lr, ud, fb

    def _check_stability(
        self,
        frame: np.ndarray,
        x: float, y: float, z: float, angle: float,
        x_tol: float, y_tol: float, z_tol: float, angle_tol: float,
        stable_counter: int,
        required_frames: int
    ) -> Tuple[int, bool]:
        """
        Check if position is stable within tolerances.
        Returns: (new_counter, is_stable)
        """
        within_tolerance = (
            abs(x) <= x_tol and
            abs(y) <= y_tol and
            abs(z) <= z_tol and
            abs(angle) <= angle_tol
        )
        
        if within_tolerance:
            new_counter = stable_counter + 1
            cv2.putText(
                frame,
                f"Stable {new_counter}/{required_frames}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            return new_counter, new_counter >= required_frames
        else:
            return 0, False

    def _draw_status(self, frame: np.ndarray, text: str, y_offset: int = 0, color=(0, 255, 255)):
        """Draw status text on frame."""
        cv2.putText(
            frame,
            text,
            (10, 90 + y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    def clip(self, v: float) -> int:
        m = self.ctx.config.MAX_RC
        return int(max(-m, min(m, v)))

    def _record_cmd(self, text: str):
        # print(text)
        self.ctx.last_cmd_texts.clear()
        self.ctx.last_cmd_texts.append(text)

    def send_rc(self, lr: float, fb: float, ud: float, yaw: float):
        lr, fb, ud, yaw = map(self.clip, (lr, fb, ud, yaw))
        
        lr = self._apply_deadzone(lr)
        fb = self._apply_deadzone(fb)
        ud = self._apply_deadzone(ud)
        yaw = self._apply_deadzone(yaw)

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
            if frame is None or frame.size == 0:
                logger.warning("Received empty frame")
                time.sleep(0.1)
                continue
                
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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

            # Battery Check (every 10 seconds)
            if self._last_battery_check == 0.0:
                self._last_battery_check = time.time()
            
            if time.time() - self._last_battery_check > 10.0:
                try:
                    battery = self.ctx.drone.get_battery()
                    if battery < 20:
                        logger.warning(f"Low battery: {battery}%")
                        cv2.putText(frame, f"LOW BATTERY: {battery}%", (10, frame.shape[0] - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self._last_battery_check = time.time()
                except Exception as e:
                    logger.error(f"Failed to check battery: {e}")

            # HUD
            y0 = 25
            if self.ctx.simulation:
                cv2.putText(
                    frame,
                    "SIMULATION (no RC sent)",
                    (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2,
                )
                y0 += 25
            else:
                cv2.putText(
                    frame,
                    f"AIRBORNE={self.ctx.airborne}",
                    (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                y0 += 25

            if ids is None:
                cv2.putText(
                    frame,
                    "NO MARKER DETECTED",
                    (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "MARKER DETECTED",
                    (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # 最近指令回饋
            if self.ctx.last_cmd_texts:
                cv2.putText(
                    frame,
                    self.ctx.last_cmd_texts[-1],
                    (10, y0 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

            # FSM
            if not self.ctx.manual_override:
                handler = self.handlers.get(self.state)
                if handler:
                    next_state = handler(self.ctx)
                    if next_state != self.state:
                        if self._validate_transition(self.state, next_state):
                            logger.info(f"[FSM] {self.state.name} -> {next_state.name}")
                            self.state = next_state
                        else:
                            logger.warning(f"[FSM] Invalid transition attempted: {self.state.name} -> {next_state.name}")

            cv2.imshow("drone", frame)

        cv2.destroyAllWindows()

    # Handlers for each state
    def handle_INIT(self, ctx: Context) -> State:
        """
        Initial state. Takeoff and transition to searching for Marker 1.
        """
        frame = ctx.frame_read.frame
        cv2.putText(
            frame,
            "INIT: Taking Off...",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        
        if not ctx.airborne and not ctx.simulation:
            ctx.drone.takeoff()
            ctx.airborne = True
            time.sleep(5.0) # Wait for takeoff
        elif ctx.simulation and not ctx.airborne:
             ctx.airborne = True
             print("[SIM] Takeoff")

        return State.TAKEOFF_SEARCH_1

    def handle_TAKEOFF_SEARCH_1(self, ctx: Context) -> State:
        """
        Search for Marker 1.
        """
        frame = ctx.frame_read.frame
        cv2.putText(
            frame,
            "STATE: SEARCH_1",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        
        target_id = 1
        poses = ctx.last_poses
        
        if target_id in poses:
            print(f"[FSM] Found Marker {target_id}, aligning...")
            self.hover()
            self.reset_pids()
            self._stable_count = 0
            return State.ALIGN_MARKER_1_INITIAL
            
        # Search pattern (e.g., rotate slowly or move up)
        # For now, just hover and wait, or move up slightly
        self.send_rc(0, 0, 20, 0) # Ascend slowly
        return State.TAKEOFF_SEARCH_1

    def handle_ALIGN_MARKER_1_INITIAL(self, ctx: Context) -> State:
        """
        Align with Marker 1.
        """
        frame = ctx.frame_read.frame
        cv2.putText(
            frame,
            "STATE: ALIGN_1",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        
        target_id = 1
        poses = ctx.last_poses
        
        if target_id not in poses:
            self.hover()
            return State.TAKEOFF_SEARCH_1 # Lost marker, go back to search
            
        rvec, tvec = poses[target_id]
        x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]
        
        # PID Control
        # Target distance: 100cm (example)
        target_dist = 100
        
        lr, ud, fb = self._update_pid_controls(ctx, x, y, z - target_dist)
        
        # Clamp speeds
        cap = 30
        lr = self._clamp_speed(lr, cap)
        ud = self._clamp_speed(ud, cap)
        fb = self._clamp_speed(fb, cap)
        
        self.send_rc(lr, fb, -ud, 0) # No yaw rotation for now
        
        # Check stability
        self._stable_count, is_stable = self._check_stability(
            frame, x, y, z - target_dist, 0,
            10, 10, 10, 100, # Tolerances (x, y, z, angle)
            self._stable_count, 10 # Required frames
        )
        
        if is_stable:
            print("[FSM] Aligned with Marker 1. Starting Move Right.")
            self.hover()
            self._state_t0 = time.time()
            return State.MOVE_RIGHT_50
            
        return State.ALIGN_MARKER_1_INITIAL

    def handle_MOVE_RIGHT_50(self, ctx: Context) -> State:
        """
        Move right 50 cm.
        Using time-based open loop control.
        Assuming speed ~20 cm/s -> 2.5 seconds.
        """
        frame = ctx.frame_read.frame
        cv2.putText(
            frame,
            "STATE: MOVE_RIGHT",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        
        move_speed = 20 # cm/s
        target_dist = 50 # cm
        move_time = target_dist / move_speed
        
        elapsed = time.time() - self._state_t0
        
        if elapsed < move_time:
            self.send_rc(move_speed, 0, 0, 0)
            cv2.putText(frame, f"Time: {elapsed:.1f}/{move_time:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return State.MOVE_RIGHT_50
        else:
            print("[FSM] Move Right Complete. Starting Line Follow.")
            self.hover()
            return State.FOLLOW_LINE

    def handle_FOLLOW_LINE(self, ctx: Context) -> State:
        """
        Follow black line until Marker 1 is seen again.
        """
        frame = ctx.frame_read.frame
        cv2.putText(
            frame,
            "STATE: FOLLOW_LINE",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        
        # Check for Marker 1
        target_id = 1
        poses = ctx.last_poses
        if target_id in poses:
             print("[FSM] Found Marker 1 again! Aligning...")
             self.hover()
             self.reset_pids()
             self._stable_count = 0
             return State.ALIGN_MARKER_1_FINAL

        # Line Detection Logic
        # 1. Convert to Gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Threshold (Black line on lighter background)
        # Invert so line is white (255) and background is black (0)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        # 3. ROI (Region of Interest) - Look at the bottom 1/3 of the screen
        h, w = thresh.shape
        roi = thresh[2*h//3:h, :]
        
        # 4. Find Contours
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (the line)
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw centroid on original frame (offset by ROI)
                cv2.circle(frame, (cx, cy + 2*h//3), 5, (0, 0, 255), -1)
                
                # Calculate Error (Center of screen vs Centroid)
                screen_center = w // 2
                error = cx - screen_center
                
                # Control
                # P-Controller for Yaw or Left/Right
                # If line is to the right (error > 0), strafe right or rotate right
                
                # Strategy: Strafe to center line, Move forward slowly
                kP = 0.2
                lr_speed = kP * error
                fb_speed = 15 # Constant forward speed
                
                lr_speed = self._clamp_speed(lr_speed, 20)
                
                self.send_rc(lr_speed, fb_speed, 0, 0)
                
                cv2.putText(frame, f"Err: {error} LR: {lr_speed:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                self.hover() # Line seen but invalid moments
        else:
            # No line found - Hover or Search (Rotate?)
            self.hover()
            cv2.putText(frame, "NO LINE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        # Display threshold for debug
        cv2.imshow("Thresh", thresh)
            
        return State.FOLLOW_LINE

    def handle_ALIGN_MARKER_1_FINAL(self, ctx: Context) -> State:
        """
        Align with Marker 1 again, then land.
        """
        frame = ctx.frame_read.frame
        cv2.putText(
            frame,
            "STATE: ALIGN_FINAL",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        
        target_id = 1
        poses = ctx.last_poses
        
        if target_id not in poses:
            self.hover()
            return State.ALIGN_MARKER_1_FINAL # Keep trying
            
        rvec, tvec = poses[target_id]
        x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]
        
        # PID Control
        target_dist = 80 # Closer for landing
        
        lr, ud, fb = self._update_pid_controls(ctx, x, y, z - target_dist)
        
        cap = 20
        lr = self._clamp_speed(lr, cap)
        ud = self._clamp_speed(ud, cap)
        fb = self._clamp_speed(fb, cap)
        
        self.send_rc(lr, fb, -ud, 0)
        
        self._stable_count, is_stable = self._check_stability(
            frame, x, y, z - target_dist, 0,
            8, 8, 8, 100,
            self._stable_count, 15
        )
        
        if is_stable:
            print("[FSM] Final Alignment Complete. Landing.")
            return State.LAND

        return State.ALIGN_MARKER_1_FINAL

    def handle_LAND(self, ctx: Context) -> State:
        """
        Land the drone.
        """
        frame = ctx.frame_read.frame
        cv2.putText(
            frame,
            "LANDING...",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        
        if not ctx.simulation:
            ctx.drone.land()
            ctx.airborne = False
            
        return State.DONE

    def handle_DONE(self, ctx: Context) -> State:
        """
        Mission complete state.
        """
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
    pid_lr.initialize()
    pid_ud.initialize()
    pid_fb.initialize()

    detector = MarkerDetector(marker_size_cm=15, calib_path="calib_tello.xml")

    ctx = Context(
        drone=drone,
        detector=detector,
        frame_read=frame_read,
        pid_lr=pid_lr,
        pid_ud=pid_ud,
        pid_fb=pid_fb,
        airborne=False,  # Ground
        simulation=True,  # Default to simulation mode
        config=DroneConfig()
    )
    fsm = DroneFSM(ctx)

    try:
        try:
            drone.send_rc_control(0, 0, 0, 0)
        except:
            pass
        fsm.run()
    except Exception as ex:
        logger.error(f"Runtime error: {ex}", exc_info=True)
    finally:
        try:
            drone.send_rc_control(0, 0, 0, 0)
        except Exception as e:
            logger.error(f"Failed to stop drone: {e}")
        try:
            drone.streamoff()
        except Exception as e:
            logger.error(f"Failed to stop stream: {e}")
        try:
            drone.end()
        except Exception as e:
            logger.error(f"Failed to end drone connection: {e}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
