# =========================
# Drone ArUco FSM with SIMULATION (no RC sent when simulate=True)
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
    ASCEND_SEARCH = enum.auto()
    CENTER_ON_TARGET = enum.auto()
    FORWARD_TO_TARGET = enum.auto()
    STRAFE_OPPOSITE = enum.auto()
    FOLLOW_MARKER_ID = enum.auto()
    PASS_UNDER_TABLE_3 = enum.auto()
    ROTATE_RIGHT_90 = enum.auto()
    ASCEND_LOCK_4 = enum.auto()
    OVERBOARD_TO_FIND_5 = enum.auto()
    ALIGN_Y5_FLIP_LAND = enum.auto()
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
        self.state = State.ASCEND_SEARCH
        self.ctx = ctx
        self.ctx = ctx
        # self.strafe_t0 = None  # Moved to below
        self.handlers = {
            State.ASCEND_SEARCH: self.handle_ASCEND_SEARCH,
            State.CENTER_ON_TARGET: self.handle_CENTER_ON_TARGET,
            State.FORWARD_TO_TARGET: self.handle_FORWARD_TO_TARGET,
            State.STRAFE_OPPOSITE: self.handle_STRAFE_OPPOSITE,
            State.FOLLOW_MARKER_ID: self.handle_FOLLOW_MARKER_ID,
            State.PASS_UNDER_TABLE_3: self.handle_PASS_UNDER_TABLE_3,
            State.ROTATE_RIGHT_90: self.handle_ROTATE_RIGHT_90,
            State.ASCEND_LOCK_4: self.handle_ASCEND_LOCK_4,
            State.OVERBOARD_TO_FIND_5: self.handle_OVERBOARD_TO_FIND_5,
            State.ALIGN_Y5_FLIP_LAND: self.handle_ALIGN_Y5_FLIP_LAND,
            State.DONE: self.handle_DONE,
        }
        self.blocking_running = False

        # State variables
        self.strafe_t0: Optional[float] = None
        
        # PASS_UNDER_TABLE_3 state variables
        self._pass_phase: int = 0
        self._pass_t0: Optional[float] = None
        self._pass_initial_height: Optional[float] = None
        self._pass_stable: int = 0
        
        # ROTATE_RIGHT_90 state variables
        self._rotate_phase: int = 0
        self._rotate_t0: Optional[float] = None
        
        # ASCEND_LOCK_4 & ALIGN_Y5_FLIP_LAND state variables
        self._ascend_stable: int = 0
        
        # OVERBOARD_TO_FIND_5 state variables
        self._over_phase: int = 0
        self._over_t0: Optional[float] = None
        
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

    def _pick_farther_id(
        self, poses: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[int]:
        # Use z (larger is farther); could also use np.linalg.norm(tvec)
        best_id, best_z = None, -1
        for mid, (_, tvec) in poses.items():
            z = float(tvec[2][0])
            if z > best_z:
                best_z, best_id = z, mid
        return best_id

    # Handlers for each state
    def handle_ASCEND_SEARCH(self, ctx: Context) -> State:
        """
        Ascend until both ID1 and ID2 are seen.
        Then pick the farther one as target and decide strafe direction.
        """
        frame = ctx.frame_read.frame
        cv2.putText(
            frame,
            "ASCEND_SEARCH",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        poses = ctx.last_poses
        if ctx.config.ID1 in poses and ctx.config.ID2 in poses and len(poses) == 2:  # seen both
            target_id = self._pick_farther_id(poses)
            non_target_id = 1 if target_id == 2 else 2
            ctx.config.TARGET_ID = target_id

            x_far = float(poses[target_id][1][0][0])
            x_near = float(poses[non_target_id][1][0][0])

            if x_far - x_near > 0:  # far is right relative to near
                ctx.config.OPPOSITE_STRAFE_SIGN = -1  # strafe left
            else:
                ctx.config.OPPOSITE_STRAFE_SIGN = +1  # strafe right
            return State.CENTER_ON_TARGET

        # According to convention: ud<0 is ascend
        self.send_rc(0, 0, ctx.config.ASCENT_SPEED, 0)
        return State.ASCEND_SEARCH

    def handle_CENTER_ON_TARGET(self, ctx: Context) -> State:
        """
        Center on the target marker (ID1 or ID2) before moving forward.
        (Currently a placeholder that immediately transitions)
        """
        frame = ctx.frame_read.frame
        cv2.putText(
            frame,
            "CENTER_ON_TARGET",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        return State.FORWARD_TO_TARGET

    def handle_FORWARD_TO_TARGET(self, ctx: Context) -> State:
        """
        Move forward towards the target marker until within distance.
        """
        frame = ctx.frame_read.frame
        cv2.putText(
            frame,
            "FORWARD_TO_TARGET",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        try:
            target_id = ctx.config.TARGET_ID
            _, tvec = ctx.last_poses[target_id]
            z = tvec[2][0]
            fb = ctx.pid_fb.update(z - ctx.config.TARGET_DIST, sleep=0.0)
            self.send_rc(0, fb, 0, 0)
            if z < ctx.config.TARGET_DIST + ctx.config.Z_TOL:
                return State.STRAFE_OPPOSITE
            return State.FORWARD_TO_TARGET
        except KeyError:  # can not find target, too close or lagging
            self.send_rc(0, 0, 0, 0)
            return State.STRAFE_OPPOSITE

    def handle_STRAFE_OPPOSITE(self, ctx: Context) -> State:
        """
        Strafe left or right (opposite to the target) for a fixed duration.
        Then transition to following the marker.
        """
        frame = ctx.frame_read.frame
        cv2.putText(
            frame,
            "STRAFE_OPPOSITE",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        if self.strafe_t0 is None:
            self.strafe_t0 = time.time()

        direction_sign = ctx.config.OPPOSITE_STRAFE_SIGN
        self.send_rc(direction_sign * ctx.config.STRAFE_SPEED, 0, 0, 0)
        if time.time() - self.strafe_t0 > ctx.config.STRAFE_TIME:
            self.strafe_t0 = None
            return State.FOLLOW_MARKER_ID
        return State.STRAFE_OPPOSITE

    def handle_FOLLOW_MARKER_ID(self, ctx: Context) -> State:
        """
        Follow the target marker (ID1 or ID2) using PID control.
        If MARKER_3 is seen, transition to PASS_UNDER_TABLE_3.
        """
        tid = ctx.config.FOLLOW_ID
        marker3 = ctx.config.MARKER_3
        target_dist = ctx.config.FOLLOW_DIS

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
        yaw_update, ud_update, fb_update = self._update_pid_controls(ctx, error_x, error_y, error_z)

        if fb_update < 0:
            fb_update = fb_update * ctx.config.FB_UPDATE_MULT
        if ud_update > 0:
            ud_update *= ctx.config.UD_UPDATE_MULT

        # Step 3: Apply speed limiting
        rot_speed = ctx.config.FOLLOW_ROT_SPE
        x_speed = ctx.config.FOLLOW_X_SPE
        y_speed = ctx.config.FOLLOW_Y_SPE
        z_speed = ctx.config.FOLLOW_Z_SPE

        yaw_update = self._clamp_speed(yaw_update, x_speed)
        ud_update = self._clamp_speed(ud_update, y_speed)
        fb_update = self._clamp_speed(fb_update, z_speed)

        # Add rotation control based on marker orientation with proper angle normalization
        # Target alignment angle - we want the marker to appear level in the image
        # 0° = horizontal alignment (marker appears level)
        # Calculate shortest angle error

        print(
            f"error x:{error_x}, error y:{error_y}, error z:{error_z}, error a:{angle_error}"   
        )
        print(f"fb err{fb_update}")

        # Dead zone - don't rotate if error is small (prevents jittering)
        angle_dead_zone = 1.0  # degrees - larger dead zone for stability
        if abs(angle_error) < angle_dead_zone:
            angle_error = 0

        # Apply speed limiting for rotation
        if angle_error > rot_speed:
            angle_error = rot_speed
        elif angle_error < -rot_speed:
            angle_error = -rot_speed

        self.send_rc(
            int(yaw_update), int(fb_update), int(-ud_update), int(-angle_error)
        )
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
        tid = ctx.config.MARKER_3

        # Initialize phase tracking (reset if coming from another state)
        # Note: In a full FSM, enter/exit actions would handle this.
        # Here we assume variables are reset when transitioning TO this state if needed,
        # or we rely on the fact that we reset them when leaving the previous state.
        # For safety, we can check if we are just entering, but for now we trust the flow.
        
        # However, to be safe against re-entry without reset:
        # We'll assume _pass_phase=0 is the start.
        
        self._draw_status(frame, f"STATE: PASS_UNDER_TABLE_3 (Phase {self._pass_phase})")

        self._draw_status(frame, f"STATE: PASS_UNDER_TABLE_3 (Phase {self._pass_phase})")

        # Phase 0: Search for MARKER_3
        if self._pass_phase == 0:
            poses = getattr(ctx, "last_poses", {}) or {}
            if tid not in poses:
                # Initialize search timer if needed
                if self._pass_t0 is None:
                    self._pass_t0 = time.time()
                
                # Check timeout
                if time.time() - self._pass_t0 > ctx.config.MAX_SEARCH_TIMEOUT:
                    logger.warning(f"Search timeout for Marker {tid}")
                    self.hover()
                    return State.DONE  # Or land/hover

                # Keep searching - gentle down movement
                search_ud = ctx.config.PASS_SEARCH_UD
                search_fb = ctx.config.PASS_SEARCH_FB
                self.send_rc(0, search_fb, search_ud, 0)
                cv2.putText(
                    frame,
                    f"Searching for Marker {tid}... ({int(ctx.config.MAX_SEARCH_TIMEOUT - (time.time() - self._pass_t0))}s)",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )
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
                self.send_rc(0, -15, 0, 0)
                cv2.putText(
                    frame,
                    f"Lost Marker {tid} during centering...",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    2,
                )
                return State.PASS_UNDER_TABLE_3

            rvec, tvec = poses[tid]
            x = float(tvec[0][0])
            y = float(tvec[1][0])
            z = float(tvec[2][0])
            z = z - ctx.config.MARKER_3_DIS
            # angle alignment
            marker_angle, _ = calculate_marker_angle(rvec)
            # normalize to [-180, 180]
            angle_err = marker_angle

            # PID for x/y
            lr, ud, fb = self._update_pid_controls(ctx, x, y, z)

            # clamp
            cap = int(ctx.config.MAX_RC)
            lr = self._clamp_speed(lr, cap)
            ud = self._clamp_speed(ud, cap)
            fb = self._clamp_speed(fb, cap)
            yaw_cmd = angle_err  # negative to reduce error

            # keep distance constant here (fb=0)
            self.send_rc(lr, fb, -ud, -yaw_cmd)

            # check tolerances (x/y and angle)
            self._pass_stable, is_stable = self._check_stability(
                frame, x, y, z, angle_err,
                ctx.config.CENTER_X_TOL, ctx.config.CENTER_Y_TOL, ctx.config.Z_TOL, ctx.config.ANGLE_TOL,
                self._pass_stable, int(ctx.config.TRACK_STABLE_N)
            )

            if not is_stable:
                self._draw_status(frame, "Centering & aligning marker 3...", y_offset=30, color=(255, 255, 0))

            if self._pass_stable >= int(ctx.config.TRACK_STABLE_N):
                print("[PASS_TABLE] Centered & aligned. Start descent")
                self.hover()
                time.sleep(0.2)
                self.reset_pids()
                self._pass_t0 = time.time()
                self._pass_phase = 2
            return State.PASS_UNDER_TABLE_3

        # Phase 2: Descend 50cm (using time-based control)
        if self._pass_phase == 2:
            DESCENT_TIME = ctx.config.PASS_DESCENT_TIME
            descent_ud = ctx.config.PASS_DESCENT_UD
            elapsed = time.time() - self._pass_t0

            if elapsed < DESCENT_TIME:
                # Continue descending (ud>0 = down)
                self.send_rc(0, 0, descent_ud, 0)
                cv2.putText(
                    frame,
                    f"Descending... {elapsed:.1f}s / {DESCENT_TIME}s",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
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
            FORWARD_TIME = ctx.config.PASS_FORWARD_TIME
            forward_fb = ctx.config.PASS_FORWARD_FB
            elapsed = time.time() - self._pass_t0

            if elapsed < FORWARD_TIME:
                # Continue moving forward
                self.send_rc(0, forward_fb, 0, 0)
                cv2.putText(
                    frame,
                    f"Moving forward... {elapsed:.1f}s / {FORWARD_TIME}s",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
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
        # Assumes _rotate_phase is reset to 0 when entering this state
        
        cv2.putText(
            frame,
            f"STATE: ROTATE_RIGHT_90 (Phase {self._rotate_phase})",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

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
            rotate_yaw = ctx.config.ROTATE_YAW
            rotate_time = ctx.config.ROTATE_TIME

            elapsed = time.time() - self._rotate_t0

            if elapsed < rotate_time:
                # Continue rotating clockwise (positive yaw)
                self.send_rc(0, 0, 0, rotate_yaw)
                cv2.putText(
                    frame,
                    f"Rotating CW... {elapsed:.1f}s / {rotate_time:.1f}s",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
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
        tid = ctx.config.MARKER_4

        # Initialize tracking
        # Assumes _ascend_stable is reset to 0 when entering this state
        
        self._draw_status(frame, f"STATE: ASCEND_LOCK_4 (find & lock ID={tid})")

        poses = getattr(ctx, "last_poses", {}) or {}

        if tid not in poses:
            # Marker not detected - continue ascending while searching
            self._ascend_stable = 0
            ascend_speed = ctx.config.ACSEND_LOCK_SPEED
            back_speed = ctx.config.BAKCWARD_LOCK_SPEED
            self.send_rc(0, back_speed, ascend_speed, 0)  # ud<0 = up
            cv2.putText(
                frame,
                f"Ascending, searching for Marker {tid}...",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )
            return State.ASCEND_LOCK_4

        # Marker detected - perform centering with PID
        rvec, tvec = poses[tid]
        x = float(tvec[0][0])
        y = float(tvec[1][0])
        z = float(tvec[2][0])

        # Display marker position
        cv2.putText(
            frame,
            f"Marker4: x={x:.1f} y={y:.1f} z={z:.1f}cm",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        # PID control for centering
        err_x = x
        err_y = y
        err_z = z - ctx.config.MARKER_4_DIS
        marker_angle, z_prime = calculate_marker_angle(rvec)
        marker_angle *= ctx.config.ANGLE_ERROR_MULT
        
        lr, ud, fb = self._update_pid_controls(ctx, err_x, err_y, err_z)
        
        # Speed limiting
        cap = int(ctx.config.MAX_RC)
        lr = self._clamp_speed(lr, cap)
        ud = self._clamp_speed(ud, cap)

        # Send control command (note: ud sign is flipped for camera-to-drone coordinate)
        self.send_rc(lr, fb, -ud, -marker_angle)

        # Check if centered within tolerance
        self._ascend_stable, is_stable = self._check_stability(
            frame, err_x, err_y, err_z, 0,
            ctx.config.CENTER_X_TOL, ctx.config.CENTER_Y_TOL, ctx.config.Z_TOL, float('inf'),
            self._ascend_stable, int(ctx.config.TRACK_STABLE_N)
        )

        if not is_stable:
            self._draw_status(frame, "Centering on marker...", y_offset=60, color=(255, 255, 0))

        # Check if stable for required number of frames
        if self._ascend_stable >= ctx.config.TRACK_STABLE_N:
            print(
                f"[ASCEND_LOCK_4] Locked on Marker {tid}, moving to OVERBOARD_TO_FIND_5"
            )
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
        tid5 = ctx.config.MARKER_5
        tid4 = ctx.config.MARKER_4

        cv2.putText(
            frame,
            f"STATE: OVERBOARD_TO_FIND_5 (ID4->up 80cm, then search ID5={tid5})",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        poses = getattr(ctx, "last_poses", {}) or {}

        # Initialize phase machine: 0=await/ascend, 1=left-move/search-5
        # Assumes _over_phase is reset to 0 when entering this state

        # Phase 0: Ensure marker 4 is seen, then ascend for configured time (approx. 80cm)
        if self._over_phase == 0:
            if tid4 not in poses:
                # Initialize wait timer if needed
                if self._over_t0 is None:
                    self._over_t0 = time.time()
                    
                # Check timeout
                if time.time() - self._over_t0 > ctx.config.MAX_SEARCH_TIMEOUT:
                    logger.warning(f"Timeout waiting for Marker {tid4}")
                    self.hover()
                    return State.DONE

                # Wait while holding hover until MARKER_4 becomes visible
                self.hover()
                cv2.putText(
                    frame,
                    f"Waiting for Marker {tid4}... ({int(ctx.config.MAX_SEARCH_TIMEOUT - (time.time() - self._over_t0))}s)",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )
                return State.OVERBOARD_TO_FIND_5
            
            # Reset timer for ascend phase if we were waiting
            if self._over_t0 is not None and (time.time() - self._over_t0 > 5.0): # Heuristic: if we waited long, reset
                 self._over_t0 = None

            ascend_ud = ctx.config.OVERBOARD_ASCEND_UD
            ascend_time = ctx.config.OVERBOARD_ASCEND_TIME

            if self._over_t0 is None:
                self._over_t0 = time.time()

            elapsed = time.time() - self._over_t0
            if elapsed < ascend_time:
                # Continue ascending (ud<0 = up)
                self.send_rc(0, 0, ascend_ud, 0)
                cv2.putText(
                    frame,
                    f"Ascending ~80cm... {elapsed:.1f}s / {ascend_time:.1f}s",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
                return State.OVERBOARD_TO_FIND_5
            else:
                # Ascend complete → move to left-move phase
                print(
                    "[OVERBOARD] Ascend complete. Begin moving left to find Marker 5."
                )
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
            lr_speed = ctx.config.OVERBOARD_LR  # Negative = left
            self.send_rc(lr_speed, 0, 0, 0)

            cv2.putText(
                frame,
                f"Moving left, searching for Marker {tid5}...",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"lr={lr_speed}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            return State.OVERBOARD_TO_FIND_5

        return State.OVERBOARD_TO_FIND_5

    def handle_ALIGN_Y5_FLIP_LAND(self, ctx: Context) -> State:
        tid = ctx.config.MARKER_5
        dis = ctx.config.MARKER_5_DIS

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

        x_m = ctx.config.MARKER_5_X_TOL
        y_m = ctx.config.MARKER_5_Y_TOL
        z_m = ctx.config.MARKER_5_Z_TOL
        
        # Check stability (using 3 frames as per original logic)
        self._ascend_stable, is_stable = self._check_stability(
            frame, error_x, error_y, error_z, 0,
            x_m, y_m, z_m, float('inf'),
            self._ascend_stable, 3
        )
        
        if is_stable:
            self.hover()
            self.ctx.drone.land()
            return State.DONE

        # Step 2: Use PID to get control outputs
        yaw_update, ud_update, fb_update = self._update_pid_controls(ctx, error_x, error_y, error_z)

        # Step 3: Apply speed limiting
        rot_speed = ctx.config.FOLLOW_ROT_SPE
        x_speed = ctx.config.FOLLOW_X_SPE
        y_speed = ctx.config.FOLLOW_Y_SPE
        z_speed = ctx.config.FOLLOW_Z_SPE

        yaw_update = self._clamp_speed(yaw_update, x_speed)
        ud_update = self._clamp_speed(ud_update, y_speed)
        fb_update = self._clamp_speed(fb_update, z_speed)

        # Add rotation control based on marker orientation with proper angle normalization
        # Target alignment angle - we want the marker to appear level in the image
        # 0° = horizontal alignment (marker appears level)
        target_alignment_angle = 0.0

        # Calculate shortest angle error
        angle_error = (marker_angle - target_alignment_angle) * ctx.config.ALIGN_ANGLE_MULT

        # Dead zone - don't rotate if error is small (prevents jittering)
        angle_dead_zone = ctx.config.ANGLE_DEAD_ZONE  # degrees - larger dead zone for stability
        if abs(angle_error) < angle_dead_zone:
            angle_error = 0

        # Apply speed limiting for rotation
        if angle_error > rot_speed:
            angle_error = rot_speed
        elif angle_error < -rot_speed:
            angle_error = -rot_speed

        self.send_rc(
            int(yaw_update), int(fb_update), int(-ud_update), int(-angle_error)
        )
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
