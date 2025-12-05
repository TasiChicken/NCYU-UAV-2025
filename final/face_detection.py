import cv2
import numpy as np
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard
import math
from djitellopy import Tello

# Object point
face_x = 15
face_y = 15


def mss(update, max_speed_threshold=30):
    if update > max_speed_threshold:
        update = max_speed_threshold
    elif update < -max_speed_threshold:
        update = -max_speed_threshold

    return update

def see_face(drone, face_cascade ):
    tvec = None
    frame_read = drone.get_frame_read()
    fs = cv2.FileStorage("calib_tello.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("K").mat()
    distortion = fs.getNode("D").mat()

    z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    x_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

    z_pid.initialize()
    y_pid.initialize()
    x_pid.initialize()
    yaw_pid.initialize()

    while True:
        frame = frame_read.frame
        # print(frame)
        if frame.sum() == 0:
            continue
        
        face_rects = face_cascade.detectMultiScale(frame, 
                                               scaleFactor=1.06,
                                               minNeighbors=20,
                                               minSize=(60, 60))
        
        for (x, y, w, h) in face_rects:
            img_pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
            obj_pts = np.array([[0, 0, 0], [face_x, 0, 0], [face_x, face_y, 0], [0, face_y, 0]], dtype=np.float32)
            _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, intrinsic, distortion)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'{tvec[2][0]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('drone', frame)
        key = cv2.waitKey(50)
        if key != -1:
            keyboard(drone, key)
        elif face_rects is not None and tvec is not None:
            (x_err, y_err, z_err) = tvec[:,0]
            z_err = z_err - 40
            x_err = x_err * 2
            y_err = - (y_err + 10) * 2

            R, err = cv2.Rodrigues(np.array([rvec[:,0]]))
            # print("err:", err)
            V = np.matmul(R, [0, 0, 1])
            rad = math.atan(V[0]/V[2])
            deg = rad / math.pi * 180
            # print(deg)
            yaw_err = yaw_pid.update(deg, sleep=0)
            
            x_err = x_pid.update(x_err, sleep=0)
            y_err = y_pid.update(y_err, sleep=0)
            z_err = z_pid.update(z_err, sleep=0)
            yaw_err = yaw_pid.update(yaw_err, sleep=0)

            print("errs:", x_err, y_err, z_err, yaw_err)
            
            xv = mss(x_err)
            yv = mss(y_err)
            zv = mss(z_err)
            rv = mss(yaw_err)
            # print(xv, yv, zv, rv)
            # drone.send_rc_control(min(20, int(xv)), min(20, int(zv//2)), min(20, int(yv//2)), 0)
            if abs(z_err) <= 15 and abs(y_err) <= 10 and abs(x_err) <= 10:
                print("Saw face!")
                cv2.destroyAllWindows()
                return
            else: 
                drone.send_rc_control(int(xv), int(zv//2), int(yv), 0)
                # print(xv, (zv//2), yv)
        else:
            drone.send_rc_control(0, 0, 0, 0)

# if __name__ == '__main__':
#     drone = Tello()
#     drone.connect()
#     drone.streamon()
    
#     see_face(drone)