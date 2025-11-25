import numpy as np
from djitellopy import tello
import cv2
import cv2.aruco as aruco

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
#me.takeoff()

cap = cv2.VideoCapture(1)

# HSV Values for BLACK line
# Black is low Value. Hue and Saturation don't matter as much, but usually low Saturation too.
# Adjust these if the black line isn't detected well.
hsvVals = [0, 0, 0, 179, 255, 60] 

sensors = 3
threshold = 0.2
width, height = 480, 360
senstivity = 3  # if number is high less sensitive
weights = [-25, -15, 0, 15, 25]
fSpeed = 15
curve = 0

# State Machine
# 0: SEARCH (Look for ArUco ID 1)
# 1: MOVE_RIGHT (Fly right until line found)
# 2: FOLLOW (Follow line)
state = 0 

def thresholding(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]])
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def getContours(imgThres, img):
    cx = 0
    found = False
    contours, hieracrhy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        biggest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(biggest)
        if area > 1000: # Minimum area to consider it a valid line
            x, y, w, h = cv2.boundingRect(biggest)
            cx = x + w // 2
            cy = y + h // 2
            cv2.drawContours(img, biggest, -1, (255, 0, 255), 7)
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            found = True
    return cx, found

def getSensorOutput(imgThres, sensors):
    imgs = np.hsplit(imgThres, sensors)
    totalPixels = (img.shape[1] // sensors) * img.shape[0]
    senOut = []
    for x, im in enumerate(imgs):
        pixelCount = cv2.countNonZero(im)
        if pixelCount > threshold * totalPixels:
            senOut.append(1)
        else:
            senOut.append(0)
        # cv2.imshow(str(x), im)
    # print(senOut)
    return senOut

def sendCommands(senOut, cx):
    global curve
    ## TRANSLATION
    lr = (cx - width // 2) // senstivity
    lr = int(np.clip(lr, -10, 10))
    if 2 > lr > -2: lr = 0

    ## Rotation - DISABLED for this task
    curve = 0
    
    # Send control: lr (roll), fSpeed (pitch), 0 (throttle), curve (yaw)
    me.send_rc_control(lr, fSpeed, 0, curve)

def findArucoMarkers(img, markerSize=4, totalMarkers=50, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use 4x4 dictionary by default, common for Tello
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)
    
    found_id_1 = False
    if ids is not None:
        if 1 in ids:
            found_id_1 = True
        if draw:
            aruco.drawDetectedMarkers(img, bboxs)
    return found_id_1

while True:
    #_, img = cap.read()
    img = me.get_frame_read().frame
    img = cv2.resize(img, (width, height))
    img = cv2.flip(img, 0)

    imgThres = thresholding(img)
    cx, line_found = getContours(imgThres, img)  ## For Translation
    
    # State Machine Logic
    if state == 0: # SEARCH
        print("State: SEARCH - Looking for ArUco ID 1")
        found_aruco = findArucoMarkers(img)
        if found_aruco:
            print("ArUco ID 1 Found! Switching to MOVE_RIGHT")
            state = 1
        else:
            # Hover while searching
            me.send_rc_control(0, 0, 0, 0)
            
    elif state == 1: # MOVE_RIGHT
        print("State: MOVE_RIGHT - Flying right, looking for line")
        # Fly right at speed 20
        me.send_rc_control(20, 0, 0, 0)
        
        # Check if line is detected
        if line_found:
            print("Line Detected! Switching to FOLLOW")
            state = 2
            
    elif state == 2: # FOLLOW
        print("State: FOLLOW - Following line")
        senOut = getSensorOutput(imgThres, sensors)
        sendCommands(senOut, cx)

    cv2.imshow("Output", img)
    cv2.imshow("Path", imgThres)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break