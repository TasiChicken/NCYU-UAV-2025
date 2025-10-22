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
    is_chessboard_found, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    is_chessboard_found, corners = cv2.findChessboardCorners(gray, (9,6), None)

    if is_chessboard_found:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)  

        cv2.drawChessboardCorners(frame, (9,6), corners2, is_chessboard_found)
        
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
is_chessboard_found, intrinsic_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
print("Camera matrix:\n", intrinsic_matrix)
print("Distortion coefficients:\n", distortion_coefficients)
# Save results
fs = cv2.FileStorage("calibration.xml", cv2.FILE_STORAGE_WRITE)
fs.write("intrinsic", intrinsic_matrix)
fs.write("distortion", distortion_coefficients)
fs.release()

print("Calibration done. Results saved in calibration.xml")
