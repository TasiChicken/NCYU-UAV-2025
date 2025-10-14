import cv2
import numpy as np

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# prepare object points for a 9x6 chessboard
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = [] # 3D points
imgpoints = [] # 2D points

# Set maximum number of calibration images
MAX_CALIBRATION_IMAGES = 4  # Adjust this value as needed

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)  

        cv2.drawChessboardCorners(frame, (9,6), corners2, ret)
        
        # Break if we have collected enough calibration images
        if len(imgpoints) >= MAX_CALIBRATION_IMAGES:
            print(f"Collected {len(imgpoints)} calibration images. Breaking...")
            break

    cv2.imshow('Calibration', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Starting calibration...")
# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)
# Save results
fs = cv2.FileStorage("calibration.xml", cv2.FILE_STORAGE_WRITE)
fs.write("intrinsic", mtx)
fs.write("distortion", dist)
fs.release()

print("Calibration done. Results saved in calibration.xml")
