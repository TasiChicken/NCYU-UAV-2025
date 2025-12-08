import cv2
from djitellopy import Tello
import time
# Import necessary functions from your final.py
# Make sure final.py is in the same directory
from final import trace_line

def mock_send_rc_control(lr, fb, ud, rot):
    """
    Mock function to replace drone.send_rc_control.
    It prints the commands instead of sending them to the drone.
    """
    print(f"RC Control: lr={lr}, fb={fb}, ud={ud}, rot={rot}")

def main():
    drone = Tello()
    drone.connect()
    drone.streamon()
    
    # Replace the real send_rc_control with our mock function
    # This prevents the drone from spinning motors if it somehow thinks it's flying,
    # and lets you see the commands in the terminal.
    drone.send_rc_control = mock_send_rc_control

    print("Battery:", drone.get_battery())
    print("Starting trace_line test (NO TAKEOFF)...")
    print("Press 'q' in the drone window to quit early if needed (or Ctrl+C in terminal)")
    
    # Sequence of trace_line calls
    print("Moving left!")
    trace_line(drone, [-8,0,0,0], [2,1,2,1,1,1,2,0,2], horizontal_trace=True, target_corner=2)
    print("1 corner detected")
    time.sleep(1)

    print("Moving up!")
    trace_line(drone, [0,0,8,0], [0,0,0,1,1,0,0,1,0], horizontal_trace=False, target_corner=3)
    print("2 corner detected")
    time.sleep(1)

    print("Moving left!")
    trace_line(drone, (-8,0,0,0), [0,1,0,0,1,1,0,0,2], horizontal_trace=True, target_corner=4)
    print("3 corner detected")
    time.sleep(1)

    print("Moving up!")
    trace_line(drone, [0,0,8,0], [2,2,2,1,1,2,2,1,2], horizontal_trace=False, target_corner=5)
    print("4 corner detected")
    time.sleep(1)

    print("Moving left!")
    trace_line(drone, (-8,0,0,0), [0,0,0,1,1,1,0,1,0], horizontal_trace=True, target_corner=6)
    print("5 corner detected")
    time.sleep(1)

    print("Moving down!")
    trace_line(drone, [0,0,8,0], [0,1,0,1,1,1,0,0,0], horizontal_trace=False, target_corner=7)
    print("6 corner detected")
    time.sleep(1)

    print("Moving left!")
    trace_line(drone, [-8,0,0,0], [0,1,0,1,1,1,2,0,2], horizontal_trace=True, target_corner=6)
    print("Target square found! Test finished.")
    
    cv2.destroyAllWindows()
    # drone.streamoff()

if __name__ == "__main__":
    main()
