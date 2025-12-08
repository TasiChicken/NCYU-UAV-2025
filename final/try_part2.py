import os.path as path
import numpy as np
import cv2
import time
import math
#import tello
from pyimagesearch.pid import PID
from djitellopy import Tello
from enum import Enum, auto

#720*960
"""
H: 190 ~280 
S:   0.68~ 1
V:   0.5~1
"""
#####################################################################
#       Const. Definition
#####################################################################
MAX_SPEED_THRESHOLD = 25
Z_BASE = np.array([[0],[0],[1]])
RC_update_para_x = 1.1 #條pid
RC_update_para_y = 1
RC_update_para_z = 1.5
RC_update_para_yaw = 1

TIME_SET_ALAPHA = 1.0
TIME_SET_BETA = 0

BASIC_VLR = 35
BASIC_VFB = 35
BASIC_VUD = 50
BASIC_YAW = 30

#TODO --->need some modification
CORRECT_THRESHOD_X = 40
CORRECT_THRESHOD_Y = 40
CORRECT_THRESHOD_Z = 17.5 #12.5
#####################################################################
#       State flags
#####################################################################

current_state = 0
class State(Enum):
    center = auto()
    up1 = auto()
    up2 = auto()
    LD = auto()

step_list = {0:'prepare',1:'step_correct',2:'step_follow_line_and_search',3:'step_finish'}

"""
step_2_follow = 4
step_2_finish = 5

step_3_1st_correct = 6
step_3_1st_ready = 7        #stop and fly right
step_3_2nd_correct = 8
step_3_2nd_ready = 9        #stop and fly left and forward

step_4_correct = 10
step_4_finish = 11 #!!!!!! No need?
"""
text = []
for i in range(100):
    str1 = str(i)+'.jpg'
    text.append(str1)
ind = 0
################################################
#   Default Chase line state and Var.
################################################
chase_LR = -1
chase_UD = 0
THRESHOLD_COUNT = 300
CHASE_LR_SPEED = 20
CHASE_UD_SPEED = 20

HSV_upper = [110,247,107]#[110,247,101]#[126,255,255]
HSV_bottom = [101,190,69]#[101,190,69]            #[97,233,232]
FRAME_DIVIDER = 5.05
GRAY_BT_THRESHOLUD = 50
GRAY_UP_THRESHOLUD = 225
#####################################################################
#       Func. Definition
#####################################################################

def keyboard(self, key):
    global is_flying
    print("key:", key)
    fb_speed = 40
    lf_speed = 40
    ud_speed = 50
    degree = 30
    if key == ord('1'):
        self.takeoff()
        is_flying = True
        print("Take off!!!")
    if key == ord('2'):
        self.land()
        is_flying = False
        print("Landed!!!")
    if key == ord('3'):
        self.send_rc_control(0, 0, 0, 0)
        print("stop!!!!")
    if key == ord('w'):
        self.send_rc_control(0, fb_speed, 0, 0)
        print("forward!!!!")
    if key == ord('s'):
        self.send_rc_control(0, (-1) * fb_speed, 0, 0)
        print("backward!!!!")
    if key == ord('a'):
        self.send_rc_control((-1) * lf_speed, 0, 0, 0)
        print("left!!!!")
    if key == ord('d'):
        self.send_rc_control(lf_speed, 0, 0, 0)
        print("right!!!!")
    if key == ord('z'):
        self.send_rc_control(0, 0, ud_speed, 0)
        print("down!!!!")
    if key == ord('x'):
        self.send_rc_control(0, 0, (-1) *ud_speed, 0)
        print("up!!!!")
    if key == ord('c'):
        self.send_rc_control(0, 0, 0, degree)
        print("rotate!!!!")
    if key == ord('v'):
        self.send_rc_control(0, 0, 0, (-1) *degree)
        print("counter rotate!!!!")
    if key == ord('5'):
        height = self.get_height()
        print(height)
    if key == ord('6'):
        battery = self.get_battery()
        print (battery)
   
def intrinsic_parameter():
    f = cv2.FileStorage('calib_tello.xml', cv2.FILE_STORAGE_READ)
    intr = f.getNode("K").mat()
    dist = f.getNode("D").mat()
    
    print("K: {}".format(intr))
    print("D: {}".format(dist))

    f.release()
    return intr, dist

###################################
#       Battery display
###################################
MIN_RC = 5
MAX_RC = 40
last_send = np.array([100, 0, 0, 0])
def send_rc(drone, lr: float, fb: float, ud: float, yaw: float):
    global last_send

    def norm(v: float) -> float:
        if v == 0:
            return 0
        av = abs(v)
        if av < MIN_RC:
            return MIN_RC if v > 0 else -MIN_RC
        elif av > MAX_RC:
            return MAX_RC if v > 0 else -MAX_RC
        else:
            return v

    lr, fb, ud, yaw = map(norm, (lr, fb, ud, yaw))

    curr = np.array([lr, fb, ud, yaw])
    diffs = np.abs(curr - last_send)
    l1 = np.sum(diffs)
    if (l1 >= 5):
        last_send = curr
        drone.send_rc_control(int(lr), int(fb), int(ud), int(yaw))

def MAX_threshold(value):
    if value > MAX_SPEED_THRESHOLD:
        print("fixed to {}".format(str(MAX_SPEED_THRESHOLD)))
        return MAX_SPEED_THRESHOLD
    elif value < -1* MAX_SPEED_THRESHOLD:
        print("fixed to {}".format(str(-1* MAX_SPEED_THRESHOLD)))
        return -1* MAX_SPEED_THRESHOLD
    else:
        return value

###################################
#       find ARUCO
###################################
counter = 0     #for waiting the drone be stable
counter_2 = 0   #to wait the drone seeing the aruco for a countinuous time(use in Step2 -> Step3)

def find_id(markerIds, id)->int:
    find_target = -1
    if markerIds is not None:
        for i in range(len(markerIds)):
            if markerIds[i,0] == id:
                find_target = i           
    return find_target

def correct_v2(drone, rvec, tvec, i, dist_diff, x_pid, y_pid, z_pid, yaw_pid, countable= True)->bool:
    global Z_BASE, counter

    PID_state = {}
    rvec_3x3,_ = cv2.Rodrigues(rvec[i])
    rvec_zbase = rvec_3x3.dot(Z_BASE)
    rx_project = rvec_zbase[0]
    rz_project = rvec_zbase[2]
    angle_diff= math.atan2(float(rz_project), float(rx_project))*180/math.pi + 90  #from -90 to 90
    #angle_diff = math.atan2(float(rx_project), float(rz_project))
    #angle_diff = math.degrees(angle_diff)
    # When angle_diff -> +90:
    #   turn counterclockwise
    # When angle_diff -> -90:
    #   turn clockwise
    #######################
    #       Z-PID
    #######################
    z_update = tvec[i,0,2] - dist_diff #70 --> 85
    PID_state["org_z"] = str(z_update)
    z_update = z_pid.update(z_update, sleep=0)
    PID_state["pid_z"] = str(z_update)

    z_update = MAX_threshold(z_update)
    
    #######################
    #       X-PID
    #######################
    x_update = tvec[i,0,0]
    PID_state["org_x"] = str(x_update)
    x_update = x_pid.update(x_update, sleep=0)
    PID_state["pid_x"] = str(x_update)
    
    x_update = MAX_threshold(x_update)
    
    #######################
    #       Y-PID
    #######################
    #y_update = tvec[i,0,1]*(-1)
    y_update = tvec[i,0,1]-10
    
    PID_state["org_y"] = str(y_update)
    y_update = y_pid.update(y_update, sleep=0)
    PID_state["pid_y"] = str(y_update)

    y_update = MAX_threshold(y_update)
    
    #######################
    #       YAW-PID
    #######################
    yaw_update = (-1)* angle_diff
    #yaw_update = angle_diff - 90
    PID_state["org_yaw"] = str(yaw_update)
    yaw_update = yaw_pid.update(yaw_update, sleep=0)
    PID_state["pid_yaw"] = str(yaw_update)

    yaw_update = MAX_threshold(yaw_update)
    
    #######################
    #   Motion Response
    #######################
    send_rc(drone, int(x_update//RC_update_para_x), int(z_update//RC_update_para_z), int(y_update//RC_update_para_y)*-1, int(yaw_update//RC_update_para_yaw))
    print("--------------------------------------------")
    #now = time.ctime()
    #print("{}: PID state".format(now))
    print("MarkerIDs: {}".format(i))
    print("tvec: {}||{}||{}||{}".format(tvec[i,0,0], tvec[i,0,1], tvec[i,0,2], angle_diff))
    print("org: {}||{}||{}||{}".format(PID_state["org_x"],PID_state["org_y"],PID_state["org_z"],PID_state["org_yaw"]))
    print("PID: {}||{}||{}||{}".format(PID_state["pid_x"],PID_state["pid_y"],PID_state["pid_z"],PID_state["pid_yaw"]))
    print("--------------------------------------------")
    #print(tvec)
    #cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1, cv2.LINE_AA)
    if(tvec[i,0,0]<=CORRECT_THRESHOD_X and tvec[i,0,0]>=(-1)*CORRECT_THRESHOD_X\
         and tvec[i,0,1]<=CORRECT_THRESHOD_Y and tvec[i,0,1]>=(-1)*CORRECT_THRESHOD_Y\
              and tvec[i,0,2]<=(dist_diff+CORRECT_THRESHOD_Z) and tvec[i,0,2]>= (dist_diff-CORRECT_THRESHOD_Z) and countable):
        counter +=1
        if counter >5:
            counter =0
            print("counter:{}".format(counter))
            return True
    else:
        print("counter:{}, neg".format(counter))
            
        counter = 0
        return False

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def check_corner(frame, f2_h, f2_w):
    # 使用 Numpy 切片定義感興趣區域 (ROI)
    # 這裡的 frame 已經是二值化後的黑白圖 (frame3)
    
    divider = int(FRAME_DIVIDER)
    w_div = int(f2_w / divider)
    h_div = int(f2_h / divider)
    
    # 定義區域 [y_start:y_end, x_start:x_end]
    roi_left = frame[:, 0:w_div]
    roi_right = frame[:, f2_w - w_div:]
    roi_up = frame[0:h_div, :]
    roi_down = frame[f2_h - h_div:, :] # 注意這裡是否需要減去更精確的值，依你原邏輯調整
    
    # 計算白色像素 (255) 的數量
    # cv2.countNonZero 比 Python 迴圈快非常多
    cnt_left = cv2.countNonZero(roi_left)
    cnt_right = cv2.countNonZero(roi_right)
    cnt_up = cv2.countNonZero(roi_up)
    cnt_down = cv2.countNonZero(roi_down)
    
    # 判斷是否超過閾值
    en_left = cnt_left >= THRESHOLD_COUNT
    en_right = cnt_right >= THRESHOLD_COUNT
    en_up = cnt_up >= THRESHOLD_COUNT
    
    # 你原本的 Down 邏輯似乎閾值不同 (500)，這裡保留原本的數值
    en_down = cnt_down >= THRESHOLD_COUNT 

    return en_up, en_down, en_left, en_right



####################################################################################
#           Main Block
####################################################################################
    
def main(drone, is_Kanahei):
    global is_flying

    # --- 初始化區塊 (只執行一次) ---
    current_state = State.center
    is_flying = True
    cali_intr, cali_dist = intrinsic_parameter()
    
    # PID 設定
    x_pid = PID(kP=0.75, kI=0.0001, kD=0.8)
    z_pid = PID(kP=0.8, kI=0.0005, kD=0.2)
    y_pid = PID(kP=0.72, kI=0.0036, kD=0.2)
    yaw_pid = PID(kP=0.8, kI=0.0001, kD=0.15)
    
    for pid in [x_pid, z_pid, y_pid, yaw_pid]:
        pid.initialize()

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    drone.streamon()

    key = -1

    frame_reader = drone.get_frame_read()

    try:
        while True: 
            print("state: {}".format(current_state))
            frame = frame_reader.frame

            height,width,_ = frame.shape
            f2_h, f2_w = int(height/3), int(width/3)
            frame2 = cv2.resize(frame,(f2_w,f2_h))
            
            frame3 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            
            gray = cv2.GaussianBlur(frame3, (5, 5), 0)

            ret, frame3 = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

            kernel = np.ones((3,3), np.uint8)
            frame3 = cv2.erode(frame3, kernel, iterations=1)
            frame3 = cv2.dilate(frame3, kernel, iterations=1)
            
            markerIds = []
            markerCorners, markerIds, rejectedCandidates =cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
            
            correct_ready = False
            
            if markerIds is not None:
                frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
                rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, cali_intr, cali_dist)
                for i in range(len(markerIds)):
                    frame = cv2.drawFrameAxes(frame, cali_intr, cali_dist, rvec[i], tvec[i], 7.5)
                print("markerIds: {}".format(markerIds) )
            ########################################
            #  Highest control for keyboard control
            ########################################
            if key!=-1:
                #send_rc(drone, 0, 0, 0, 0)
                keyboard(drone,key)
            ##########################################################################
            #       FSM states
            ##########################################################################
            elif(current_state == State.center):
                target_idex = find_id(markerIds, 1)
                if target_idex != -1:
                    correct_ready = correct_v2(drone, rvec, tvec, target_idex, 55,x_pid, y_pid, z_pid, yaw_pid)
                else:     
                    send_rc(drone, 0,-20,0,0)
                    correct_ready = False

                if (correct_ready == True):
                    current_state = State.up1
                    last_en_up = False
                    start_time = time.time()

            elif(current_state == State.up1):

                if(is_Kanahei):
                    if(time.time() - start_time > 0.8): #fly left  for 0.8sec
                        current_state = State.up2
                        last_en_up = False
                    else:
                        send_rc(drone, -1*(CHASE_LR_SPEED),0,0,0)

                else:
                    en_up,en_down,en_left,en_right = check_corner(frame3,f2_h,f2_w)

                    indicator_color = (0, 255, 0)  # 綠色 (B, G, R)
                    thickness = 30
                    if en_up:
                        cv2.rectangle(frame, (0, 0), (width, thickness), indicator_color, -1)
                    if en_down:
                        cv2.rectangle(frame, (0, height - thickness), (width, height), indicator_color, -1)
                    if en_left:
                        cv2.rectangle(frame, (0, 0), (thickness, height), indicator_color, -1)
                    if en_right:
                        cv2.rectangle(frame, (width - thickness, 0), (width, height), indicator_color, -1)

                    if(en_up):
                        send_rc(drone, 0,0,CHASE_UD_SPEED,0)
                    elif(en_left):
                        send_rc(drone, -1*(CHASE_LR_SPEED),0,0,0)

                    if(last_en_up and not en_up and en_left):
                        current_state = State.up2
                        last_en_up = False
                    else:
                        last_en_up = en_up
            elif(current_state == State.up2):
                en_up,en_down,en_left,en_right = check_corner(frame3,f2_h,f2_w)

                indicator_color = (0, 255, 0)  # 綠色 (B, G, R)
                thickness = 30
                if en_up:
                    cv2.rectangle(frame, (0, 0), (width, thickness), indicator_color, -1)
                if en_down:
                    cv2.rectangle(frame, (0, height - thickness), (width, height), indicator_color, -1)
                if en_left:
                    cv2.rectangle(frame, (0, 0), (thickness, height), indicator_color, -1)
                if en_right:
                    cv2.rectangle(frame, (width - thickness, 0), (width, height), indicator_color, -1)

                if(en_up):
                    send_rc(drone, 0,0,CHASE_UD_SPEED,0)
                elif(en_left):
                    send_rc(drone, -1*(CHASE_LR_SPEED),0,0,0)

                if(last_en_up and not en_up and en_left):
                    current_state = State.LD
                    last_en_up = False
                else:
                    last_en_up = en_up
            elif(current_state == State.LD):
                target_idex = find_id(markerIds, 2)
                if target_idex != -1:
                    correct_ready = correct_v2(drone, rvec, tvec, target_idex, 55,x_pid, y_pid, z_pid, yaw_pid)
                else:     
                    correct_ready = False
                    en_up,en_down,en_left,en_right = check_corner(frame3,f2_h,f2_w)

                    indicator_color = (0, 255, 0)  # 綠色 (B, G, R)
                    thickness = 30
                    if en_up:
                        cv2.rectangle(frame, (0, 0), (width, thickness), indicator_color, -1)
                    if en_down:
                        cv2.rectangle(frame, (0, height - thickness), (width, height), indicator_color, -1)
                    if en_left:
                        cv2.rectangle(frame, (0, 0), (thickness, height), indicator_color, -1)
                    if en_right:
                        cv2.rectangle(frame, (width - thickness, 0), (width, height), indicator_color, -1)


                    if(en_down):
                        send_rc(drone, 0,0,CHASE_UD_SPEED,0)
                    elif(en_left):
                        send_rc(drone, -1*(CHASE_LR_SPEED),0,0,0)
                    else :
                        send_rc(drone, 0,-10,0,0)

                if (correct_ready == True):
                    return

            else:
                print("state: out of state")
                ################################
                #   Floating if no instructions
                ################################
                if is_flying == True:
                    send_rc(drone, 0, 0, 0, 0)      #Stop in the air
                #now = time.ctime()
                print("No instructions")
            #send_rc(drone, 0,0,0,0)
            cv2.imshow('frame', frame)
            #cv2.imshow('filter', frame2)
            cv2.imshow('filter-bw', frame3)


            key = cv2.waitKey(1)
            #send_rc(drone, 0, 0, 0, 0)

            if key == ord('q'):
                cv2.imwrite(text[ind], frame)
                ind+=1
                print('shot: {}'.format(ind))

    except KeyboardInterrupt:
        #cap.release()
        print("fail to open film")
        cv2.destroyAllWindows()