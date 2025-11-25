import time
import enum
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from djitellopy import Tello
import numpy as np
import cv2
from pyimagesearch.pid import PID

MIN_RC = 10
MAX_RC = 50

class FlightState(enum.Enum):
    IDLE = enum.auto()
    TAKEOFF_SEARCH = enum.auto()
    ALIGN_START    = enum.auto()
    FOLLOW_LINE    = enum.auto()
    ALIGN_END_LAND = enum.auto()

class MarkerDetector:
    def __init__(self, marker_size_cm: float, calib_path="calib_tello.xml"):
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
            [[-half_marker_size,  half_marker_size, 0],
             [ half_marker_size,  half_marker_size, 0],
             [ half_marker_size, -half_marker_size, 0],
             [-half_marker_size, -half_marker_size, 0]],
            dtype=np.float32
        )

    def detect(self, gray):
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
                    # 多存 corners 方便算中心點
                    poses[int(mid)] = (rvec, tvec, corners[i])
        return ids, poses

drone = None
detector = None
frame_read = None

def send_rc(lr: float, fb: float, ud: float, yaw: float):
    global drone

    def norm(v: float) -> float:
        if v == 0:
            return 0.0
        av = abs(v)
        if av < MIN_RC:
            return MIN_RC if v > 0 else -MIN_RC
        elif av > MAX_RC:
            return MAX_RC if v > 0 else -MAX_RC
        else:
            return v

    lr, fb, ud, yaw = map(norm, (lr, fb, ud, yaw))
    drone.send_rc_control(int(lr), int(fb), int(ud), int(yaw))

def get_marker_center(corners):
    # corners shape: (1,4,2)
    pts = corners[0]
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    return cx, cy

def line_follow_control(frame, pid_lr, forward_speed=20):
    """
    只做左右修正 + 固定向前速度的追線控制
    回傳 (lr, fb)
    """
    h, w, _ = frame.shape
    roi_start = int(h * 2 / 3)
    roi = frame[roi_start:h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 黑色線在亮背景 → 用 INV
    _, binary = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lr_cmd = 0
    fb_cmd = 0

    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.drawContours(roi, [largest], -1, (0, 255, 0), 2)
            cv2.circle(roi, (cx, cy), 5, (0, 0, 255), -1)

            center_x = w // 2
            error_x = cx - center_x

            lr_cmd = -pid_lr.update(error_x, sleep=0)
            fb_cmd = forward_speed

            cv2.putText(frame, f"line_err={error_x}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        lr_cmd = 0
        fb_cmd = 0

    frame[roi_start:h, :] = roi

    # 只有在 FOLLOW_LINE 才顯示 binary
    cv2.imshow("binary", binary)

    return lr_cmd, fb_cmd

# ======================
# 各狀態 handler function
# ======================

def handle_takeoff_search(frame,
                          marker_visible: bool,
                          pid_lr_marker, pid_ud_marker, pid_fb_marker
                          ) -> FlightState:
    cv2.putText(frame, "STATE: TAKEOFF_SEARCH", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if marker_visible:
        pid_lr_marker.initialize()
        pid_ud_marker.initialize()
        pid_fb_marker.initialize()
        send_rc(0, 0, 0, 0)
        return FlightState.ALIGN_START
    else:
        # 緩慢上升找 marker
        send_rc(0, 0, 20, 0)
        return FlightState.TAKEOFF_SEARCH

def handle_align_start(frame,
                       marker_visible: bool,
                       ids, poses,
                       center_x: int, center_y: int,
                       pid_lr_marker, pid_ud_marker, pid_fb_marker,
                       pid_lr_line,
                       desired_dist_cm: float,
                       marker_visible_prev: bool
                       ) -> Tuple[FlightState, bool]:
    cv2.putText(frame, "STATE: ALIGN_START", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if not marker_visible:
        send_rc(0, 0, 0, 0)
        return FlightState.TAKEOFF_SEARCH, marker_visible_prev

    mid = int(ids.flatten()[0])
    rvec, tvec, corners = poses[mid]
    cx, cy = get_marker_center(corners)

    dist_z = float(tvec[2])

    err_x = cx - center_x
    err_y = cy - center_y
    err_z = dist_z - desired_dist_cm

    lr_cmd = -pid_lr_marker.update(err_x, sleep=0)
    ud_cmd = -pid_ud_marker.update(err_y, sleep=0)
    fb_cmd = -pid_fb_marker.update(err_z, sleep=0)

    send_rc(lr_cmd, fb_cmd, ud_cmd, 0)

    cv2.putText(frame, f"ex={err_x} ey={err_y} ez={err_z}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 對準完成，進入追線
    if (abs(err_x) < 20) and (abs(err_y) < 20) and (abs(err_z) < 20):
        pid_lr_line.initialize()
        send_rc(0, 0, 0, 0)
        return FlightState.FOLLOW_LINE, True   # 進入 FOLLOW_LINE 時，marker 一開始在畫面裡
    else:
        return FlightState.ALIGN_START, marker_visible_prev

def handle_follow_line(frame,
                       marker_visible: bool,
                       marker_visible_prev: bool,
                       pid_lr_line,
                       pid_lr_marker, pid_ud_marker, pid_fb_marker
                       ) -> Tuple[FlightState, bool]:
    cv2.putText(frame, "STATE: FOLLOW_LINE", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 追線控制
    lr_cmd, fb_cmd = line_follow_control(frame, pid_lr_line)
    send_rc(lr_cmd, fb_cmd, 0, 0)

    new_state = FlightState.FOLLOW_LINE
    new_marker_prev = marker_visible_prev

    # 判斷離開 / 再次進入 marker
    if (not marker_visible) and marker_visible_prev:
        new_marker_prev = False
    elif marker_visible and (not marker_visible_prev):
        # 從沒看到 → 再次看到，當成終點 marker
        new_marker_prev = True
        pid_lr_marker.initialize()
        pid_ud_marker.initialize()
        pid_fb_marker.initialize()
        new_state = FlightState.ALIGN_END_LAND

    return new_state, new_marker_prev

def handle_align_end_land(frame,
                          marker_visible: bool,
                          ids, poses,
                          center_x: int, center_y: int,
                          pid_lr_marker, pid_ud_marker, pid_fb_marker,
                          desired_dist_cm: float
                          ) -> Tuple[FlightState, bool]:
    cv2.putText(frame, "STATE: ALIGN_END_LAND", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if not marker_visible:
        send_rc(0, 0, 0, 0)
        return FlightState.ALIGN_END_LAND, False

    mid = int(ids.flatten()[0])
    rvec, tvec, corners = poses[mid]
    cx, cy = get_marker_center(corners)
    dist_z = float(tvec[2])

    err_x = cx - center_x
    err_y = cy - center_y
    err_z = dist_z - desired_dist_cm

    lr_cmd = -pid_lr_marker.update(err_x, sleep=0)
    ud_cmd = -pid_ud_marker.update(err_y, sleep=0)
    fb_cmd = -pid_fb_marker.update(err_z, sleep=0)

    send_rc(lr_cmd, fb_cmd, ud_cmd, 0)

    cv2.putText(frame, f"ex={err_x} ey={err_y} ez={err_z}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 對準足夠好 → 降落
    if (abs(err_x) < 15) and (abs(err_y) < 15) and (abs(err_z) < 15):
        send_rc(0, 0, 0, 0)
        time.sleep(0.5)
        drone.land()
        return FlightState.ALIGN_END_LAND, True

    return FlightState.ALIGN_END_LAND, False


def loop(pid_lr_marker, pid_ud_marker, pid_fb_marker, pid_lr_line):
    global drone, detector, frame_read

    state = FlightState.TAKEOFF_SEARCH
    marker_visible_prev = False
    desired_dist_cm = 60.0

    while True:
        key = cv2.waitKey(1)
        if key == ord("1"):
            drone.takeoff()
            time.sleep(1.0)
            print("[KEY] takeoff")
            state = FlightState.IDLE
        elif key == ord("2"):
            drone.land()
            print("[KEY] land")
            return
        

        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w, _ = frame.shape
        center_x = w // 2
        center_y = h // 2

        ids, poses = detector.detect(gray)
        marker_visible = ids is not None and len(ids) > 0

        cv2.line(frame, (center_x, 0), (center_x, h), (255, 0, 0), 1)
        cv2.line(frame, (0, center_y), (w, center_y), (255, 0, 0), 1)

        if state == FlightState.IDLE:
            continue
        if state == FlightState.TAKEOFF_SEARCH:
            state = handle_takeoff_search(
                frame, marker_visible,
                pid_lr_marker, pid_ud_marker, pid_fb_marker
            )
        elif state == FlightState.ALIGN_START:
            state, marker_visible_prev = handle_align_start(
                frame, marker_visible, ids, poses,
                center_x, center_y,
                pid_lr_marker, pid_ud_marker, pid_fb_marker,
                pid_lr_line,
                desired_dist_cm,
                marker_visible_prev,
            )
        elif state == FlightState.FOLLOW_LINE:
            state, marker_visible_prev = handle_follow_line(
                frame, marker_visible, marker_visible_prev,
                pid_lr_line,
                pid_lr_marker, pid_ud_marker, pid_fb_marker
            )
        elif state == FlightState.ALIGN_END_LAND:
            state, landed = handle_align_end_land(
                frame, marker_visible, ids, poses,
                center_x, center_y,
                pid_lr_marker, pid_ud_marker, pid_fb_marker,
                desired_dist_cm
            )
            if landed:
                break

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    global drone, detector, frame_read

    drone = Tello()
    detector = MarkerDetector(marker_size_cm=15, calib_path="calib_tello.xml")

    drone.connect()
    print(f"Battery: {drone.get_battery()}%")

    drone.streamon()
    time.sleep(2.0)
    frame_read = drone.get_frame_read()

    drone.takeoff()
    time.sleep(2.0)

    pid_lr_marker = PID(kP=0.5, kI=0.0, kD=0.0)
    pid_ud_marker = PID(kP=0.5, kI=0.0, kD=0.0)
    pid_fb_marker = PID(kP=0.4, kI=0.0, kD=0.0)
    pid_lr_line   = PID(kP=0.2, kI=0.0, kD=0.0)

    pid_lr_marker.initialize()
    pid_ud_marker.initialize()
    pid_fb_marker.initialize()
    pid_lr_line.initialize()

    try:
        send_rc(0, 0, 0, 0)
        loop(pid_lr_marker, pid_ud_marker, pid_fb_marker, pid_lr_line)
    except Exception as ex:
        print(ex)
    finally:
        try:
            send_rc(0, 0, 0, 0)
            drone.streamoff()
            drone.end()
            cv2.destroyAllWindows()
        except Exception as ex:
            print(ex)

if __name__ == "__main__":
    main()
