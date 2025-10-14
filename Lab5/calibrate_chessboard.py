# calibrate_chessboard.py
import cv2, numpy as np, time, os, glob
from pathlib import Path
from djitellopy import Tello

pattern_size = (9, 6)
square_size = 2.5

def collect_and_calibrate():
    out_dir = Path("images")
    out_dir.mkdir(parents=True, exist_ok=True)

    tello = Tello()
    tello.connect()
    tello.streamon()
    frame_read = tello.get_frame_read()

    print("S=存圖  C=開始校正  Q=離開")
    saved = 0
    K, D = None, None

    while True:
        frame = frame_read.frame
        if frame is None:
            time.sleep(0.01)
            continue

        disp = frame.copy()
        cv2.putText(disp, f"Saved: {saved}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Tello - Capture chessboard", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('s'), ord('S')):
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = str(out_dir / f"cb_{ts}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
            print("saved:", path)
        elif key in (ord('c'), ord('C')):
            print("start calibration...")
            K, D = run_calibration(str(out_dir))
            if K is not None:
                save_xml("calib_tello.xml", K, D)
                print("Saved to calib_tello.xml")
            else:
                print("Calibration failed. Need more/better images.")
        elif key in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyAllWindows()
    try:
        tello.streamoff()
    except Exception:
        pass
    tello.end()

def run_calibration(img_dir):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                  glob.glob(os.path.join(img_dir, "*.png")))
    if len(imgs) == 0:
        print("No images found in", img_dir)
        return None, None

    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints, imgpoints = [], []
    h, w = None, None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    used = 0
    for p in imgs:
        img = cv2.imread(p)
        if img is None: 
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if h is None:
            h, w = gray.shape[:2]
        ret, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )
        if not ret:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), crit)
        objpoints.append(objp)
        imgpoints.append(corners)
        used += 1

    if used < 10:
        print(f"Only {used} valid boards found. Capture more diverse views (>=15).")
        return None, None

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    tot_err = 0.0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        tot_err += err
    mean_err = tot_err / len(objpoints)

    print("RMS:", rms)
    print("mean reprojection error:", mean_err)
    print("K:\n", K)
    print("D:\n", D.ravel())

    return K, D

def save_xml(path, K, D):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("K", K)
    fs.write("D", D)
    fs.release()

if __name__ == "__main__":
    collect_and_calibrate()
