import argparse
import cv2
import numpy as np
import sys
import os
from collections import deque
DEFAULT_WEBCAM_INDEX = 0
DEFAULT_INTRINSIC    = (500.0, 666.67, 320.0, 240.0)  
DEFAULT_DISTORTION   = (0.0, 0.0, 0.0, 0.0, 0.0)       
DEFAULT_FACE_W_MM    = 160.0
DEFAULT_FACE_H_MM    = 200.0
DEFAULT_FACE_BS      = 0.9  
DEFAULT_PERSON_H_MM  = 1700.0
DEFAULT_PERSON_W_MM  = 500.0
DEFAULT_PERSON_BS    = 0.90
DEFAULT_CASCADE_PATH = "/mnt/data/haarcascade_frontalface_default.xml"
def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--webcam", type=int, default=DEFAULT_WEBCAM_INDEX, help="攝影機 index")
    ap.add_argument("--intrinsic", type=float, nargs=4, metavar=("fx","fy","cx","cy"),
                    default=list(DEFAULT_INTRINSIC), help="相機內參")
    ap.add_argument("--dist", type=float, nargs=5, metavar=("k1","k2","p1","p2","k3"),
                    default=list(DEFAULT_DISTORTION), help="畸變係數")
    ap.add_argument("--cascade", type=str, default=DEFAULT_CASCADE_PATH, help="Haar 模型路徑")
    ap.add_argument("--face-width-mm", type=float, default=DEFAULT_FACE_W_MM, help="臉寬 (mm)")
    ap.add_argument("--face-height-mm", type=float, default=DEFAULT_FACE_H_MM, help="臉高 (mm)")
    ap.add_argument("--face-bbox-scale", type=float, default=DEFAULT_FACE_BS, help="臉框縮放比例")
    ap.add_argument("--person-width-mm", type=float, default=DEFAULT_PERSON_W_MM, help="人寬 (mm)")
    ap.add_argument("--person-height-mm", type=float, default=DEFAULT_PERSON_H_MM, help="人高 (mm)")
    ap.add_argument("--person-bbox-scale", type=float, default=DEFAULT_PERSON_BS, help="人框縮放比例")
    ap.add_argument("--show-hog", action="store_true", help="顯示 HOG 人框(未縮放)供除錯")
    ap.add_argument("--draw-axes", action="store_true", default=False, help="繪製座標軸")
    ap.add_argument("--no-smooth", action="store_true", help="關閉距離移動平均平滑")
    return ap.parse_args()
def load_cascade(path):
    if not os.path.exists(path):
        here = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(here, os.path.basename(path))
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"Haar cascade 不存在: {path}")
    return cv2.CascadeClassifier(path)
def build_planar_obj_points(width_mm, height_mm):
    w2, h2 = width_mm/2.0, height_mm/2.0
    objp = np.array([
        [-w2, -h2, 0.0],  
        [ w2, -h2, 0.0],  
        [ w2,  h2, 0.0],  
        [-w2,  h2, 0.0],  
    ], dtype=np.float32)
    return objp
def rect_corners_from_bbox(bbox):
    x, y, w, h = bbox
    pts = np.array([
        [x,     y+h],
        [x+w,   y+h],
        [x+w,   y  ],
        [x,     y  ],
    ], dtype=np.float32)
    return pts
def shrink_bbox(bbox, scale, W, H):
    x, y, w, h = bbox
    cx, cy = x + w/2.0, y + h/2.0
    nw, nh = w*scale, h*scale
    x1 = int(round(cx - nw/2.0)); y1 = int(round(cy - nh/2.0))
    x2 = int(round(cx + nw/2.0)); y2 = int(round(cy + nh/2.0))
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
    return (x1, y1, x2-x1, y2-y1)
def solve_pnp_distance(objp, img_pts, K, dist):
    ok, rvec, tvec = cv2.solvePnP(objp, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    z_m = float(tvec[2][0]) / 1000.0
    d_m = float(np.linalg.norm(tvec)) / 1000.0
    return rvec, tvec, z_m, d_m
def draw_axes(img, K, dist, rvec, tvec, axis_len=150.0):
    try:
        cv2.drawFrameAxes(img, K, dist, rvec, tvec, axis_len)
    except Exception:
        pass
def moving_average(q, new, maxlen=10):
    q.append(new)
    if len(q) > maxlen:
        q.popleft()
    return sum(q)/len(q)
def main():
    args = parse_args()
    fx, fy, cx, cy = args.intrinsic
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    dist = np.array(args.dist, dtype=np.float64)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    face_cascade = load_cascade(args.cascade)
    cap = cv2.VideoCapture(args.webcam)
    if not cap.isOpened():
        print(f"無法開啟攝影機 {args.webcam}", file=sys.stderr)
        sys.exit(1)
    face_z_q, person_z_q = deque(maxlen=8), deque(maxlen=8)
    print("按 ESC 離開。")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        face_info_txt = "face: NA"
        if len(faces) > 0:
            x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
            if args.face_bbox_scale < 1.0:
                x, y, w, h = shrink_bbox((x,y,w,h), args.face_bbox_scale, W, H)
            objp_f = build_planar_obj_points(args.face_width_mm, args.face_height_mm)
            imgp_f = rect_corners_from_bbox((x,y,w,h))
            pose_f = solve_pnp_distance(objp_f, imgp_f, K, dist)
            if pose_f is not None:
                rvec_f, tvec_f, z_m_f, d_m_f = pose_f
                if not args.no_smooth:
                    z_m_f = moving_average(face_z_q, z_m_f, maxlen=8)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                face_info_txt = f"face {d_m_f:.2f}m"
                cv2.putText(frame, face_info_txt, (x, max(0, y-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                if args.draw_axes:
                    draw_axes(frame, K, dist, rvec_f, tvec_f, axis_len=100.0)
        rects, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
        person_info_txt = "person: NA"
        if len(rects) > 0:
            x, y, w, h = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)[0]
            if args.show_hog:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)  
            if args.person_bbox_scale < 1.0:
                x, y, w, h = shrink_bbox((x,y,w,h), args.person_bbox_scale, W, H)
            objp_p = build_planar_obj_points(args.person_width_mm, args.person_height_mm)
            imgp_p = rect_corners_from_bbox((x,y,w,h))
            pose_p = solve_pnp_distance(objp_p, imgp_p, K, dist)
            if pose_p is not None:
                rvec_p, tvec_p, z_m_p, d_m_p = pose_p
                if not args.no_smooth:
                    z_m_p = moving_average(person_z_q, z_m_p, maxlen=8)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,165,255), 2)  
                person_info_txt = f"person {d_m_p:.2f}m"
                cv2.putText(frame, person_info_txt, (x, min(H-10, y+h+20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
                if args.draw_axes:
                    draw_axes(frame, K, dist, rvec_p, tvec_p, axis_len=200.0)
        cv2.putText(frame, face_info_txt + "  |  " + person_info_txt,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("HOG person + Haar face + SolvePnP distance", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
