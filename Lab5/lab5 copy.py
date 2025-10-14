import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID

def main():
    drone = Tello()
    drone.connect()
    drone.streamon()
    frame_read = drone.get_frame_read()
    yaw_pid = PID(kP=0.8, kI=0.0, kD=0.15)
    ud_pid = PID(kP=0.8, kI=0.0, kD=0.10)
    fb_pid = PID(kP=0.6, kI=0.0, kD=0.10)
    yaw_pid.initialize()
    ud_pid.initialize()
    fb_pid.initialize()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters_create()
    marker_size = 15
    fs = cv2.FileStorage("calib_tello.xml", cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("K").mat()
    dist_coeffs = fs.getNode("D").mat()
    fs.release()
    target_dist = 80.0
    while True:
        frame = frame_read.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
            #for i in range(len(ids)):
                #cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 10)
            rvec, tvec = rvecs[0][0], tvecs[0][0]
            x, y, z = tvec[0], tvec[1], tvec[2]
            distance = math.sqrt(x**2 + y**2 + z**2)
            cv2.putText(frame, f"x:{x:.2f} y:{y:.2f} z:{z:.2f} dist:{distance:.2f}cm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            yaw_out = yaw_pid.update(x, sleep=0.0)
            ud_out = ud_pid.update(y, sleep=0.0)
            fb_out = fb_pid.update(z - target_dist, sleep=0.0)
            drone.send_rc_control(0, int(-fb_out), int(ud_out), int(yaw_out))
        cv2.imshow("drone", frame)
        key = cv2.waitKey(33)
        if key == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
