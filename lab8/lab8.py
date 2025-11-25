import argparse
import cv2
import numpy as np
import sys
import os
from typing import List, Tuple

DEFAULT_WEBCAM_INDEX = 0
DEFAULT_INTRINSIC    = (500.0, 666.67, 320.0, 240.0)  # fx fy cx cy for 640x480
DEFAULT_DISTORTION   = (0.0, 0.0, 0.0, 0.0, 0.0)

DEFAULT_FACE_W_MM    = 180.0
DEFAULT_FACE_H_MM    = 200.0
DEFAULT_FACE_BS      = 0.90

DEFAULT_PERSON_W_MM  = 600.0
DEFAULT_PERSON_H_MM  = 1800.0
DEFAULT_PERSON_BS    = 0.90

DEFAULT_CASCADE_PATH = "./haarcascade_frontalface_default.xml"

# ---------- 參數解析 ----------
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
    ap.add_argument("--draw-axes", action="store_true", help="繪製座標軸")
    ap.add_argument("--max-faces", type=int, default=10, help="臉最大顯示數量")
    ap.add_argument("--max-persons", type=int, default=10, help="人最大顯示數量（NMS 後）")
    return ap.parse_args()

# ---------- 工具 ----------
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
    return np.array([
        [-w2, -h2, 0.0],
        [ w2, -h2, 0.0],
        [ w2,  h2, 0.0],
        [-w2,  h2, 0.0],
    ], dtype=np.float32)

def rect_corners_from_bbox(bbox):
    x, y, w, h = bbox
    return np.array([
        [x,     y+h],
        [x+w,   y+h],
        [x+w,   y  ],
        [x,     y  ],
    ], dtype=np.float32)

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

def draw_axes(img, K, dist, rvec, tvec, axis_len=120.0):
    try:
        cv2.drawFrameAxes(img, K, dist, rvec, tvec, axis_len)
    except Exception:
        pass

def nms_boxes(rects: List[Tuple[int,int,int,int]], overlapThresh: float=0.65):
    """簡易 NMS，輸入 (x,y,w,h) 列表，輸出過濾後列表。"""
    if len(rects) == 0:
        return []
    boxes = np.array(rects, dtype=np.float32)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    pick = []
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        suppress = [last]
        for pos in range(0, len(idxs)-1):
            i = idxs[pos]
            xx1 = max(x1[last], x1[i])
            yy1 = max(y1[last], y1[i])
            xx2 = min(x2[last], x2[i])
            yy2 = min(y2[last], y2[i])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            overlap = (w * h) / areas[i]
            if overlap > overlapThresh:
                suppress.append(i)
        idxs = np.delete(idxs, np.isin(idxs, suppress).nonzero()[0])
    return [tuple(map(int, boxes[i])) for i in pick]

# ---------- 主流程 ----------
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

    print("按 ESC 離開。")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 臉部多目標
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        faces = sorted(list(faces), key=lambda r: r[2]*r[3], reverse=True)[:args.max_faces]
        objp_face = build_planar_obj_points(args.face_width_mm, args.face_height_mm)
        for i, (x, y, w, h) in enumerate(faces):
            if args.face_bbox_scale < 1.0:
                x, y, w, h = shrink_bbox((x,y,w,h), args.face_bbox_scale, W, H)
            imgp = rect_corners_from_bbox((x,y,w,h))
            pose = solve_pnp_distance(objp_face, imgp, K, dist)
            if pose is None:
                continue
            rvec, tvec, z_m, d_m = pose
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"face#{i+1} {d_m:.2f}m",
                        (x, max(0, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
            if args.draw_axes:
                draw_axes(frame, K, dist, rvec, tvec, axis_len=90.0)

        # 行人多目標（HOG + NMS）
        rects, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
        rects = nms_boxes(rects, overlapThresh=0.65)
        rects = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)[:args.max_persons]
        objp_person = build_planar_obj_points(args.person_width_mm, args.person_height_mm)
        for j, (x, y, w, h) in enumerate(rects):
            if args.person_bbox_scale < 1.0:
                x, y, w, h = shrink_bbox((x,y,w,h), args.person_bbox_scale, W, H)
            imgp = rect_corners_from_bbox((x,y,w,h))
            pose = solve_pnp_distance(objp_person, imgp, K, dist)
            if pose is None:
                continue
            rvec, tvec, z_m, d_m = pose
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,165,255), 2)
            cv2.putText(frame, f"person#{j+1} {d_m:.2f}m",
                        (x, min(H-5, y+h+18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,165,255), 2)
            if args.draw_axes:
                draw_axes(frame, K, dist, rvec, tvec, axis_len=140.0)

        cv2.imshow("Multi-target: HOG person + Haar face + SolvePnP", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
