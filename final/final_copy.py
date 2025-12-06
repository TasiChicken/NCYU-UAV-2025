import cv2 
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard
from face_detection import see_face
from object_detection import detect_objects


black_thres = 30

sq = {
    "tl":0, "tm":1, "tr":2, "ml":3, "mm":4, "mr":5, "bl":6, "bm":7, "br":8
}

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def square_same(input, truth):
    for i in range(9):
        if truth[i] == 2:
            input[i] = truth[i]
    return input == truth

def line_follower(frame):
    frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
    height, width = frame.shape
    
    # 計算九宮格的邊界
    h_borders = [0, height//3, 2*height//3, height]
    w_borders = [0, width//3, 2*width//3, width]
    
    # 初始化九宮格的計數器
    squares = {
        'tl': 0, 'tm': 0, 'tr': 0,
        'ml': 0, 'mm': 0, 'mr': 0,
        'bl': 0, 'bm': 0, 'br': 0
    }
    
    # 計算每個格子的總像素數
    total_pixels = (height//3) * (width//3)
    threshold = 0.1  # 可以調整這個閾值
    max_ratio = 0
    # 遍歷每個九宮格
    for i, (h1, h2) in enumerate(zip(h_borders[:-1], h_borders[1:])):
        for j, (w1, w2) in enumerate(zip(w_borders[:-1], w_borders[1:])):
            # 取得當前格子的所有像素
            region = frame[h1:h2, w1:w2]
            # 計算黑色像素（值為0）的數量
            black_pixels = np.sum(region == 0)
            # 計算黑色像素的比例
            black_ratio = black_pixels / total_pixels
            max_ratio = max(max_ratio, black_ratio)
            
            # 根據位置設置對應的格子值
            pos = ['tl', 'tm', 'tr',
                  'ml', 'mm', 'mr',
                  'bl', 'bm', 'br'][i*3 + j]
            squares[pos] = 1 if black_ratio > threshold else 0
    
    # 返回結果列表，保持原有的順序
    return [squares['tl'], squares['tm'], squares['tr'],
            squares['ml'], squares['mm'], squares['mr'],
            squares['bl'], squares['bm'], squares['br']], max_ratio

def put_detected_square(frame, detected_squares, is_gray):
    height, width = 0, 0
    if is_gray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, gray = cv2.threshold(gray, black_thres, 255, cv2.THRESH_BINARY)
        height, width = gray.shape
        frame = np.dstack((gray, gray, gray))
    else:
        (height, width, _) = frame.shape
    w_mid = int(width/2)
    h_mid = int(height/2)

    x_list = [150, w_mid - 30, width - 180]
    y_list = [150, h_mid - 30, height - 180]

    for i, detected in enumerate(detected_squares):
        x = x_list[i % 3]
        y = y_list[int(i / 3)]
        if detected:
            cv2.putText(frame, text=f'black', fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, org=(x, y), color=(0, 0, 255), thickness=2)
        else:
            cv2.putText(frame, text=f'white', fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, org=(x, y), color=(255, 0, 0), thickness=2)
        
    return frame

def trace_line(drone, speed_output, target_square, horizontal_trace=False, target_corner=None):
    detected_squares = [0,0,0,0,0,0,0,0,0]
    
    # Create a window to display the target square grid
    grid_size = 300  # Size of the grid window
    cell_size = grid_size // 3  # Size of each cell in the grid

    while not square_same(detected_squares, target_square):
        frame = drone.get_frame_read().frame
        (height, width, _) = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, gray = cv2.threshold(gray, black_thres, 255, cv2.THRESH_BINARY)

        detected_squares, black_ratio = line_follower(gray)
        #print(black_ratio)
        frame = put_detected_square(frame, detected_squares, True)
        frame = cv2.putText(frame, text=f'{target_corner}', org=(width//2, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=2)
        cv2.imshow("drone", frame)

        # Create a blank image for the grid
        grid_image = np.ones((grid_size, grid_size, 3), dtype=np.uint8) * 255  # Start with a white grid

        # Draw the grid based on target_square
        for i in range(3):
            for j in range(3):
                color = (255, 255, 255)  # Default to white
                if target_square[i * 3 + j] == 1:
                    color = (0, 0, 0)  # Black
                elif target_square[i * 3 + j] == 2:
                    color = (128, 128, 128)  # Gray

                # Draw the rectangle
                top_left = (j * cell_size, i * cell_size)
                bottom_right = ((j + 1) * cell_size, (i + 1) * cell_size)
                cv2.rectangle(grid_image, top_left, bottom_right, color, -1)

        # Display the grid
        cv2.imshow("Target Square Grid", grid_image)

        key = cv2.waitKey(33)
        if key != -1:
            keyboard(drone, key)
        else:
            lr, fb, ud, rot = speed_output
            if horizontal_trace and detected_squares[:3] == [1,1,1]:
                ud += 0
            elif horizontal_trace and  detected_squares[-3:] == [1,1,1]:
                ud -= 0
            elif not horizontal_trace and detected_squares[::3] == [1,1,1]:
                lr -= 0
            elif not horizontal_trace and detected_squares[2::3] == [1,1,1]:
                lr += 0

            if detected_squares == [0,0,0,0,0,0,0,0,0]:
                fb -= 10
            #else:
                #fb += int(mss((0.3 - black_ratio) * 60, 10)) 
            drone.send_rc_control(lr, fb, ud, rot)
    drone.send_rc_control(0,0,0,0)

def mss(update, max_speed_threshold=30):
    if update > max_speed_threshold:
        update = max_speed_threshold
    elif update < -max_speed_threshold:
        update = -max_speed_threshold

    return update

def init_pids(kP=0.5, kI=0.0001, kD=0.1):
    z_pid = PID(kP=kP, kI=kI, kD=kD)
    y_pid = PID(kP=kP, kI=kI, kD=kD)
    x_pid = PID(kP=kP, kI=kI, kD=kD)
    yaw_pid = PID(kP=kP, kI=kI, kD=kD)

    z_pid.initialize()
    y_pid.initialize()
    x_pid.initialize()
    yaw_pid.initialize()
    return z_pid, y_pid, x_pid, yaw_pid

def see(drone, markId):
    frame_read = drone.get_frame_read()

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    fs = cv2.FileStorage("calib_tello.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("K").mat()
    distortion = fs.getNode("D").mat()

    z_pid, y_pid, x_pid, yaw_pid = init_pids()

    while True:
        frame = frame_read.frame
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        print(markerIds)
        
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        cv2.imshow('drone', frame)
        key = cv2.waitKey(33)
        if key != -1:
            keyboard(drone, key)
        elif markerIds is not None:
            # Find the index of markId in markerIds
            target_idx = None
            for i, id in enumerate(markerIds):
                if id[0] == markId:
                    target_idx = i
            if target_idx is None:
                continue

            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)
            (x_err, y_err, z_err) = tvec[target_idx][0]
            z_err = z_err - 50
            x_err = x_err * 2
            y_err = - (y_err + 10) * 2

            R, err = cv2.Rodrigues(np.array([rvec[target_idx]]))
            # print("err:", err)
            V = np.matmul(R, [0, 0, 1])
            rad = math.atan(V[0]/V[2])
            deg = rad / math.pi * 180
            
            x_err = x_pid.update(x_err, sleep=0)
            y_err = y_pid.update(y_err, sleep=0)
            z_err = z_pid.update(z_err, sleep=0)
            yaw_err = yaw_pid.update(deg*2, sleep=0)

            print("errs:", x_err, y_err, z_err, yaw_err)
            
            xv = mss(x_err)
            yv = mss(y_err)
            zv = mss(z_err)
            rv = int(mss(yaw_err, 50))
            # print(xv, yv, zv, rv)
            # drone.send_rc_control(min(20, int(xv//2)), min(20, int(zv//2)), min(20, int(yv//2)), 0)
            if abs(z_err) <= 10 and abs(y_err) <= 50 and abs(x_err) <= 50 and abs(yaw_err) <= 10:
                print("Saw marker", markId)
                cv2.destroyAllWindows()
                return
            else: 
                drone.send_rc_control(int(xv), int(zv/1.5), int(yv), int(rv))
        
        else:
            drone.send_rc_control(0, 0, 0, 0)

# See the multiple markers with the same markId (using x coor to compare), follow the nearest one 
def see_multi(drone, markId, z_dist=60):
    frame_read = drone.get_frame_read()
    # cap = cv2.VideoCapture(0)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    fs = cv2.FileStorage("calib_tello.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("K").mat()
    distortion = fs.getNode("D").mat()

    z_pid, y_pid, x_pid, yaw_pid = init_pids()

    while True:
        frame = frame_read.frame
        # ret, frame = cap.read()
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        print(markerIds)
        
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        key = cv2.waitKey(33)
        if key != -1:
            keyboard(drone, key)

        elif markerIds is not None:
            # Find the index of markId in markerIds
            target_idxes = []
            for i, id in enumerate(markerIds):
                if id[0] == markId:
                    target_idxes.append(i)
                    
            if not target_idxes:  # No target found
                continue

            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)

            # Find the nearest marker and follow it
            target_idx = target_idxes[0]
            if type(target_idxes) == list:
                min_x_dist = float('Inf')
                for idx in target_idxes:
                    x_err = abs(tvec[idx][0][0])  ### TODO: calibration or using distance
                    if x_err < min_x_dist:
                        min_x_dist = x_err
                        target_idx = idx
                    
                    # Put x_err of each marker on the frame
                    text_coor = (np.sum(markerCorners[idx][0], axis=0) / 4).tolist()
                    text_coor = tuple([int(i) for i in text_coor])
                    text_coor = (text_coor[0], text_coor[1] + 25 * (i+1))
                    print(text_coor)    
                    cv2.putText(frame, text=f'idx: {idx}, x_err: {round(x_err, 2)}',
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, 
                                org=text_coor, color=(0, 255, 0), thickness=1)

            # Display the nearest marker
            frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec[target_idx], tvec[target_idx], 7)

            (x_err, y_err, z_err) = tvec[target_idx][0]
            z_err = z_err - z_dist
            x_err = x_err * 2
            y_err = - (y_err + 10) * 2

            R, err = cv2.Rodrigues(np.array([rvec[target_idx]]))
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
            # drone.send_rc_control(min(20, int(xv//2)), min(20, int(zv//2)), min(20, int(yv//2)), 0)
            if abs(z_err) <= 10 and abs(y_err) <= 50 and abs(x_err) <= 50:
                print("Saw marker", markId)
                # cap.release()
                cv2.destroyAllWindows()
                return
            else: 
                if abs(y_err) >= 10 or abs(x_err) >= 10:
                    drone.send_rc_control(int(xv), 0, int(yv), 0)
                else:
                    drone.send_rc_control(int(xv), int(zv), int(yv), 0)
        else:
            drone.send_rc_control(0, 0, 0, 0)

        cv2.imshow('drone', frame)

def test(drone):
    print("Moving down!")
    trace_line(drone, [0,0,-15,0], [1,2,0,0,0,0,2,2,2], False)
    print("7 corner detected")

    print("Moving left! Going through the table")
    trace_line(drone, [-20,0,0,0], [0,2,2,0,2,2,0,0,0], False)
    print("8 corner detected")

    print("Moving up!")
    trace_line(drone, [0,0,20,0], [0,0,0,1,1,0,0,1,0], False)
    print("9 corner detected")

    print("Moving left!")
    trace_line(drone, (-20,0,0,0), [0,2,0,0,1,1,0,0,0], True)
    print("10 corner detected")
    see(drone, 2)
    drone.rotate_clockwise(180)
    drone.move("forward", 70)
    detected_doll = detect_objects(drone)
    if detected_doll == "Kanahei":
        drone.move("left", 50)
    else:
        drone.move("right", 50)
    
    drone.move("forward", 100)
    see(drone, 3)
    drone.land()

def main():
    drone = Tello()
    drone.connect()
    drone.streamon()
    drone.takeoff()
    time.sleep(4)
    
    # # 1. 飛到人臉前，飛過板子，看到第二張人臉，飛過桌子底下
    # drone.move("up", 75)
    # drone.move("forward", 140)
    # see_face(drone, face_cascade)
    # drone.move("up", 60)
    # drone.move("forward", 130)
    # drone.move("down", 150)
    # drone.move("forward", 130)

    # 2. 偵測娃娃，開始循線
    drone.move("up", 30)

    detected_doll = detect_objects(drone)
    print(f"Saw {detect_objects}\n")
    see(drone, 1)
    drone.move("left", 20)
    if detected_doll == "Kanahei":

        drone.move("left", 100)

        print("Move left!")
        trace_line(drone, [-8,0,0,0], [0,1,0,1,1,1,0,0,0], horizontal_trace=True, target_corner=0)
        print("0 corner")

        print("Move up!")
        trace_line(drone, [0,0,8,0], [0,0,0,1,1,0,0,1,0], horizontal_trace=False, target_corner=1)
        print("1 corner")

        print("Move left!")
        trace_line(drone, [-8,0,0,0], [0,0,0,0,1,1,0,1,2], horizontal_trace=True, target_corner=2)
        print("2 corner")

        print("Move down!")
        trace_line(drone, [0,0,-8,0], [0,1,0,1,1,0,0,0,0], horizontal_trace=False, target_corner=3)
        print("3 corner")

        print("Moving left!")
        trace_line(drone, [-8,0,0,0],  [0,0,0,0,1,1,0,1,2], horizontal_trace=True, target_corner=4)
        print("4 corner")

        print("Move down!")
        trace_line(drone, [0,0,8,0], [0,1,0,1,1,0,0,0,0], horizontal_trace=False, target_corner=5)
        print("5 corner")

        print("Move left!")
        trace_line(drone, [-8,0,0,0], [0,1,0,1,1,1,2,0,2], horizontal_trace=True, target_corner=6)

    else:
        print("Moving left!")
        trace_line(drone, [-8,0,0,0], [2,1,2,1,1,1,2,0,2], horizontal_trace=True, target_corner=2)
        print("1 corner detected")

        print("Moving up!")
        trace_line(drone, [0,0,8,0], [0,0,0,1,1,0,0,1,0], horizontal_trace=False, target_corner=3)
        print("2 corner detected")

        print("Moving left!")
        trace_line(drone, (-8,0,0,0), [0,1,0,0,1,1,0,0,2], horizontal_trace=True, target_corner=4)
        print("3 corner detected")

        print("Moving up!")
        trace_line(drone, [0,0,8,0], [2,2,2,1,1,2,2,1,2], horizontal_trace=False, target_corner=5)
        print("4 corner detected")

        print("Moving left!")
        trace_line(drone, (-8,0,0,0), [0,0,0,1,1,1,0,1,0], horizontal_trace=True, target_corner=6)
        print("5 corner detected")

        print("Moving down!")
        trace_line(drone, [0,0,8,0], [0,1,0,1,1,1,0,0,0], horizontal_trace=False, target_corner=7)


        
    # TODO part3 and part4
     #rotate 180 degree
     #find human face and pass the table from below

     #rotate 90 degree
     #tell the doll and decide the marker to center
     #center on the marker and land

     # ------------------------------------------------------
    # Part 3: 看到 Marker 2 -> 轉身 180 -> 看人臉 -> 過桌子
    # ------------------------------------------------------
    print("--- Starting Part 3 ---")
    
    # 1. 確保先對準 Marker 2，作為轉身的基準點
    # 稍微後退一點避免太貼牆，好讓鏡頭看得到 marker
    drone.move("back", 20) 
    see(drone, 2)
    
    # 2. 旋轉 180 度面向桌子
    print("Rotating 180...")
    drone.rotate_clockwise(180)
    time.sleep(1) # 等待機身穩定

    # 3. 對準人臉 (假設 see_face 會調整無人機位置對齊人臉)
    print("Aligning with Face...")
    see_face(drone, face_cascade)
    
    # 4. 往下降並穿越桌子
    # 假設人臉在桌子上，無人機需要下降夠多才能鑽過去 (例如下降 60-80cm)
    print("Going under the table...")
    drone.move("down", 60) 
    # 穿越桌子的距離 (根據地圖桌寬 + 緩衝，約 150-200cm)
    drone.move("forward", 180) 

    # ------------------------------------------------------
    # Part 4: 轉身 90 -> 辨識娃娃 -> 左右分流 -> 對準 Marker 3 降落
    # ------------------------------------------------------
    print("--- Starting Part 4 ---")

    # 1. 穿過桌子後，根據地圖路線需左轉 90 度 (面向終點區)
    print("Rotating 90 CCW...")
    drone.rotate_counter_clockwise(90)
    time.sleep(1)

    # 2. 辨識娃娃
    print("Detecting Doll...")
    # 稍微上升一點或前進一點以利辨識 (視鏡頭角度而定，如果太低可能要 move up)
    drone.move("up", 20) 
    target_doll = detect_objects(drone)
    print(f"Decision: Doll is {target_doll}")

    # 3. 根據娃娃決定左右分流 (假設 Kanahei 往左側桌子，其他往右側桌子)
    if target_doll == "Kanahei":
        print("Go Left path")
        drone.move("left", 60)
    else:
        print("Go Right path")
        drone.move("right", 60)

    # 4. 前進接近降落區
    print("Approaching Landing Zone...")
    drone.move("forward", 100)

    # 5. 搜尋並對準 Marker 3 (使用 see_multi 會找最近的那個 ID 3)
    print("Aligning with Marker 3...")
    see_multi(drone, 3, 50)

    # 6. 降落
    print("Landing...")
    drone.land()
    
if __name__ == "__main__":
    # test()
    main()

