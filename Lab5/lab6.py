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
    
    # Alternative method using the marker's apparent orientation
    # The marker's local X-axis direction in camera coordinates  
    marker_x_in_camera = R @ np.array([1, 0, 0])
    
    # Project onto image plane (XY plane of camera)
    image_x = marker_x_in_camera[0]
    image_y = marker_x_in_camera[1] 
    
    # Calculate angle relative to horizontal axis in image
    angle_rad_alt = math.atan2(image_y, image_x)
    angle_deg_alt = math.degrees(angle_rad_alt)
    
    # Use the alternative method as it's more intuitive for image-plane rotation
    return angle_deg_alt, R

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
    fb_pid = PID(kP=0.4, kI=0.0, kD=0.0)    # Start simple for forward/back
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        manual_control_active = False  # Reset each loop iteration
        
        # KEYBOARD CONTROL CHECK - HIGHEST PRIORITY (check first)
        key = cv2.waitKey(33)
        if key == 27:
            break
        if key != -1:
            manual_control_used = keyboard(drone, key)
            manual_control_active = manual_control_used
        

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
                
                # Display position and distance information
                cv2.putText(frame, f"Pos x:{x:.1f} y:{y:.1f} z:{z:.1f} dist:{distance:.1f}cm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, f"Error x:{error_x:.1f} y:{error_y:.1f} z:{error_z:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                cv2.putText(frame, f"Target Distance: {target_dist}cm", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                cv2.putText(frame, f"Marker Angle: {marker_angle:.1f} deg", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                
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
                
                # Normalize angle error to [-180, 180] range for shortest rotation path
                while angle_error > 180:
                    angle_error -= 360
                while angle_error < -180:
                    angle_error += 360
                
                # Also check if we're closer to ±90° alignment (vertical orientation)
                angle_error_90 = marker_angle - 90.0
                while angle_error_90 > 180:
                    angle_error_90 -= 360
                while angle_error_90 < -180:
                    angle_error_90 += 360
                
                angle_error_270 = marker_angle - 270.0
                while angle_error_270 > 180:
                    angle_error_270 -= 360
                while angle_error_270 < -180:
                    angle_error_270 += 360
                
                # Choose the smallest angle error (closest alignment)
                if abs(angle_error_90) < abs(angle_error):
                    angle_error = angle_error_90
                    target_alignment_angle = 90.0
                if abs(angle_error_270) < abs(angle_error):
                    angle_error = angle_error_270  
                    target_alignment_angle = 270.0
                
                # Dead zone - don't rotate if error is small (prevents jittering)
                angle_dead_zone = 10.0  # degrees - larger dead zone for stability
                if abs(angle_error) < angle_dead_zone:
                    rotate_out = 0
                    alignment_status = f"ALIGNED({target_alignment_angle:.0f}°)"
                else:
                    rotate_out = angle_error * 0.2  # Even gentler rotation control
                    alignment_status = f"ROTATING->({target_alignment_angle:.0f}°)"
                
                # Apply speed limiting for rotation
                if rotate_out > max_speed_threshold:
                    rotate_out = max_speed_threshold
                elif rotate_out < -max_speed_threshold:
                    rotate_out = -max_speed_threshold
                
                # Display PID outputs for tuning
                cv2.putText(frame, f"PID yaw:{-yaw_update:.1f} ud:{-ud_update:.1f} fb:{fb_update:.1f} rot:{rotate_out:.1f}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                cv2.putText(frame, f"Angle Error: {angle_error:.1f}° Status: {alignment_status}", (10, 175),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
                # Step 4: Send control commands for automatic tracking (6 directions + rotation)
                # Only send if manual control is not active AND it's been at least 100ms since last command
                if not manual_control_active:
                    # Send the speed-limited commands to drone (rc left/right forward/back up/down yaw)
                    # NOTE: fb_update is already correct - positive means move forward, negative means move backward
                    # FIXED: ud_update needs to be flipped - when marker is below (positive y), drone should go DOWN (negative command)
                    drone.send_rc_control(int(yaw_update*2), int(fb_update*2), int(-ud_update*2), int(rotate_out*5))
                    
                    # Print message indicating drone is actively following the pattern
                    print("🎯 DRONE IS FOLLOWING THE PATTERN! 🎯")
                    print(f"📍 Position - x:{x:.1f}cm, y:{y:.1f}cm, z:{z:.1f}cm, distance:{distance:.1f}cm")
                    print(f"📐 Marker angle: {marker_angle:.1f}° | Angle error: {angle_error:.1f}° | Status: {alignment_status}")
                    print(f"🚁 Auto tracking - Error x:{error_x:.1f} y:{error_y:.1f} z:{error_z:.1f} -> Commands yaw:{int(yaw_update)} fb:{int(fb_update)} ud:{int(-ud_update)} rot:{int(rotate_out)}")
                    print("─" * 60)  # Separator line
                else:
                    # Pattern is detected but manual control is active
                    print("👀 Pattern detected but manual control is active - Auto tracking paused")
                

        else:
            # NO MARKER DETECTED - Reset PID controllers to prevent drift
            yaw_pid.initialize()
            ud_pid.initialize() 
            fb_pid.initialize()
            
            # Send hover command to stop any movement
            if not manual_control_active:
                drone.send_rc_control(0, 0, 0, 0)
                # Print no pattern message
                print("❌ No ArUco pattern detected - Drone hovering in place")
            
        # Display control status
        if manual_control_active:
            cv2.putText(frame, "MANUAL CONTROL OVERRIDE", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        elif ids is not None:
            cv2.putText(frame, "AUTO TRACKING ACTIVE", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(frame, "NO MARKER DETECTED", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        
        # Show the frame
        cv2.imshow("drone", frame)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
