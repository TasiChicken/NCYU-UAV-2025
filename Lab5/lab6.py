import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID

def calculate_marker_angle(rvec):
    """
    Calculate the in-plane rotation angle of the marker (rotation around camera's optical axis)
    This represents how much the marker appears rotated in the camera image
    0° = marker aligned with image horizontal/vertical axes
    """
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Extract the rotation angle around the Z-axis (optical axis of camera)
    # This is the "roll" angle that we see as rotation in the image plane
    
    # Method 1: Extract from rotation matrix using atan2
    # R[0,0] and R[1,0] represent the marker's X-axis projected onto camera XY plane
    angle_rad = math.atan2(-R[1, 0], R[0, 0])
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg, R

def keyboard(drone, key, last_command_time=None):
    #global is_flying
    print("key:", key)
    fb_speed = 40
    lf_speed = 40
    ud_speed = 50
    degree = 30
    
    # Track if this is a movement command that should override auto tracking
    is_movement_command = False
    
    if key == ord('1'):
        drone.takeoff()
        #is_flying = True
    elif key == ord('2'):
        drone.land()
        #is_flying = False
    elif key == ord('3'):
        drone.send_rc_control(0, 0, 0, 0)
        print("stop!!!!")
        is_movement_command = True
    elif key == ord('w'):
        drone.send_rc_control(0, fb_speed, 0, 0)
        print("forward!!!!")
        is_movement_command = True
    elif key == ord('s'):
        drone.send_rc_control(0, (-1) * fb_speed, 0, 0)
        print("backward!!!!")
        is_movement_command = True
    elif key == ord('a'):
        drone.send_rc_control((-1) * lf_speed, 0, 0, 0)
        print("left!!!!")
        is_movement_command = True
    elif key == ord('d'):
        drone.send_rc_control(lf_speed, 0, 0, 0)
        print("right!!!!")
        is_movement_command = True
    elif key == ord('z'):
        drone.send_rc_control(0, 0, ud_speed, 0)
        print("down!!!!")
        is_movement_command = True
    elif key == ord('x'):
        drone.send_rc_control(0, 0, (-1) * ud_speed, 0)
        print("up!!!!")
        is_movement_command = True
    elif key == ord('c'):
        drone.send_rc_control(0, 0, 0, degree)
        print("rotate!!!!")
        is_movement_command = True
    elif key == ord('v'):
        drone.send_rc_control(0, 0, 0, (-1) * degree)
        print("counter rotate!!!!")
        is_movement_command = True
    
    # Return True only if a movement command was used (should override auto tracking)
    return is_movement_command

def main():
    drone = Tello()
    drone.connect()
    
    # Check battery level before takeoff
    battery_level = drone.get_battery()
    print(f"Battery level: {battery_level}%")
    if battery_level < 20:
        print("WARNING: Battery level too low for takeoff!")
    
    drone.streamon()
    frame_read = drone.get_frame_read()

    # Step 1: Start with I=0, D=0 for initial tuning
    # Step 2: Tune P until drone stops near target distance
    # Step 3: Add I to reduce steady-state error (but may cause oscillation)
    # Step 4: Add D to reduce oscillation and smooth response
    
    yaw_pid = PID(kP=0.5, kI=0.0, kD=0.0)   # Start simple for yaw (left/right)
    ud_pid = PID(kP=0.5, kI=0.0, kD=0.0)    # Start simple for up/down
    fb_pid = PID(kP=0.5, kI=0.0, kD=0.0)    # Start simple for forward/back
    
    yaw_pid.initialize()
    ud_pid.initialize()
    fb_pid.initialize()
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    marker_size = 15
    fs = cv2.FileStorage("calib_tello.xml", cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("K").mat()
    dist_coeffs = fs.getNode("D").mat()
    fs.release()
    target_dist = 80.0


    manual_control_active = False  # Track manual control state
    
    while True:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        manual_control_active = False  # Reset each loop iteration
        
        # KEYBOARD CONTROL CHECK - HIGHEST PRIORITY (check first)
        key = cv2.waitKey(33)
        if key == 27:
            break
        if key != -1:
            manual_control_active = keyboard(drone, key)        

        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)
        
        # ONLY send tracking commands when marker is detected AND manual control is not active
        if ids is not None:
            print(f"ArUco marker detected! ID(s): {ids.flatten()}")
            
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            # Use the newer API for pose estimation
            rvecs = []
            tvecs = []
            for i in range(len(corners)):
                # Define 3D points of ArUco marker corners in marker coordinate system
                objPoints = np.array([
                    [-marker_size/2, marker_size/2, 0],
                    [marker_size/2, marker_size/2, 0],
                    [marker_size/2, -marker_size/2, 0],
                    [-marker_size/2, -marker_size/2, 0]
                ], dtype=np.float32)
                
                # Estimate pose for each marker
                success, rvec, tvec = cv2.solvePnP(objPoints, corners[i], camera_matrix, dist_coeffs)
                if success:
                    rvecs.append(rvec)
                    tvecs.append(tvec)
                    # Draw coordinate axes for visualization (15cm length)
                    # Suppress warnings about axes endpoints being out of frame
                    try:
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.15)
                    except cv2.error:
                        pass  # Ignore OpenCV errors when axes are out of frame bounds
            
            if len(rvecs) > 0:
                rvec, tvec = rvecs[0], tvecs[0]
                x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]
                distance = math.sqrt(x**2 + y**2 + z**2)
                
                # Calculate marker rotation angle using rvec
                marker_angle, z_prime = calculate_marker_angle(rvec)
                
                # Calculate errors for PID tuning analysis
                error_x = x  # Left/Right error
                error_y = y  # Up/Down error
                error_z = z - target_dist  # Forward/Back error (target 80cm)
                
                
                # PID calculations for automatic tracking (6 DOF control)
                # Step 1: Calculate raw error values
                print(f"org_x: {error_x}")
                print(f"org_y: {error_y}")  
                print(f"org_z: {error_z}")
                
                # Step 2: Use PID to get control outputs
                yaw_update = yaw_pid.update(error_x, sleep=0.0)      # Left/Right movement
                ud_update = ud_pid.update(error_y, sleep=0.0)        # Up/Down movement
                fb_update = fb_pid.update(error_z, sleep=0.0)        # Forward/Back movement
                
                # Print PID outputs
                print(f"pid_x: {yaw_update}")
                print(f"pid_y: {ud_update}")
                print(f"pid_z: {fb_update}")
                
                # Step 3: Apply speed limiting to prevent loss of control (建議限制最高速度防止失控)
                max_speed_threshold = 25
                
                # Limit yaw (left/right) speed
                if yaw_update > max_speed_threshold:
                    yaw_update = max_speed_threshold
                elif yaw_update < -max_speed_threshold:
                    yaw_update = -max_speed_threshold
                
                # Limit up/down speed  
                if ud_update > max_speed_threshold:
                    ud_update = max_speed_threshold
                elif ud_update < -max_speed_threshold:
                    ud_update = -max_speed_threshold
                
                # Limit forward/back speed
                if fb_update > max_speed_threshold:
                    fb_update = max_speed_threshold
                elif fb_update < -max_speed_threshold:
                    fb_update = -max_speed_threshold
                
                # Add rotation control based on marker orientation with proper angle normalization
                # Target alignment angle - we want the marker to appear level in the image
                # 0° = horizontal alignment (marker appears level)
                target_alignment_angle = 0.0
                
                # Calculate shortest angle error  
                angle_error = marker_angle - target_alignment_angle
                
                # Dead zone - don't rotate if error is small (prevents jittering)
                angle_dead_zone = 5.0  # degrees - larger dead zone for stability
                if abs(angle_error) < angle_dead_zone:
                    angle_error = 0
                
                # Apply speed limiting for rotation
                if angle_error > max_speed_threshold:
                    angle_error = max_speed_threshold
                elif angle_error < -max_speed_threshold:
                    angle_error = -max_speed_threshold
                
                if not manual_control_active:
                    drone.send_rc_control(int(yaw_update), int(fb_update), int(-ud_update), int(-angle_error))
                    print(f"Sent rc control -> left_right: {yaw_update}, forward_back: {fb_update}, up_down: {-ud_update}, rotate: {-angle_error}")
        # Show the frame
        cv2.imshow("drone", frame)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
