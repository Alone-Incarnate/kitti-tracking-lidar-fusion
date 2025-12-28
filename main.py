import cv2
import numpy as np
import os
from ultralytics import YOLO


# ===========================
# CONFIG
# ===========================
SEQ = "0000"

IMG_DIR = f"project/data/KITTI/{SEQ}/image_02"
LIDAR_DIR = f"project/data/KITTI/{SEQ}/velodyne"
CALIB_PATH = f"project/data/KITTI/{SEQ}/calib.txt"

OUTPUT_VIDEO = "output.mp4"
FPS = 15


# ===========================
# TRACKER (NO INSTALL)
# ===========================
class SimpleTracker:
    def __init__(self, max_lost=10, iou_thresh=0.3):
        self.next_id = 0
        self.tracks = {}  # id -> {"bbox":[], "lost":int}
        self.max_lost = max_lost
        self.iou_thresh = iou_thresh

    def iou(self, b1, b2):
        x1,y1,x2,y2 = b1
        a1,b1_,a2,b2_ = b2

        xx1 = max(x1,a1)
        yy1 = max(y1,b1_)
        xx2 = min(x2,a2)
        yy2 = min(y2,b2_)

        if xx2 <= xx1 or yy2 <= yy1:
            return 0.0

        inter = (xx2-xx1)*(yy2-yy1)
        area1 = (x2-x1)*(y2-y1)
        area2 = (a2-a1)*(b2_-b1_)

        return inter / float(area1 + area2 - inter)

    def update(self, detections):
        # detections: list of [x1,y1,x2,y2,label]
        updated = {}

        # increment lost counters
        for tid in self.tracks:
            self.tracks[tid]["lost"] += 1

        # match detections to tracks
        for det in detections:
            box = det[:4]
            label = det[4]

            best_tid = None
            best_iou = 0

            for tid, trk in self.tracks.items():
                i = self.iou(box, trk["bbox"])
                if i > best_iou:
                    best_iou = i
                    best_tid = tid

            if best_iou > self.iou_thresh:
                self.tracks[best_tid]["bbox"] = box
                self.tracks[best_tid]["label"] = label
                self.tracks[best_tid]["lost"] = 0
            else:
                self.tracks[self.next_id] = {
                    "bbox": box,
                    "label": label,
                    "lost": 0
                }
                self.next_id += 1

        # keep only active
        for tid, trk in self.tracks.items():
            if trk["lost"] <= self.max_lost:
                updated[tid] = trk

        self.tracks = updated

        return [(tid, trk["bbox"], trk["label"]) for tid, trk in self.tracks.items()]


# ===========================
# CALIBRATION
# ===========================
def read_calib(path):
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                key,val = line.split(":",1)
            else:
                key,val = line.split(" ",1)
            data[key.strip()] = np.array([float(x) for x in val.split()])
    return data


calib = read_calib(CALIB_PATH)
P2 = calib["P2"].reshape(3,4)
R  = calib["R_rect"].reshape(3,3)
Tr = calib["Tr_velo_cam"].reshape(3,4)


# ===========================
# MODEL + TRACKER
# ===========================
model = YOLO("yolov8n.pt")
tracker = SimpleTracker()


# ===========================
# FILE LISTING
# ===========================
files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
files = sorted(files)

print(f"Found {len(files)} frames")


# ===========================
# VIDEO WRITER
# ===========================
first = cv2.imread(os.path.join(IMG_DIR, files[0]))
h,w = first.shape[:2]

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (w,h)
)


# ===========================
# MAIN LOOP
# ===========================
ALLOWED = {0: "Pedestrian", 2: "Car"}  # strict filter

for fname in files:

    frame_id = int(fname.split(".")[0])

    img_path = os.path.join(IMG_DIR, fname)
    lidar_path = os.path.join(LIDAR_DIR, f"{frame_id:06d}.bin")

    # ---- load image ----
    img = cv2.imread(img_path)

    # ---- load lidar ----
    pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1,4)
    pts = pts[:, :3]

    # ---- project lidar ----
    pts_h = np.hstack((pts, np.ones((pts.shape[0],1))))
    pts_cam = R @ (Tr @ pts_h.T)

    z = pts_cam[2]
    valid = z > 0
    pts_cam = pts_cam[:, valid]
    z = z[valid]

    pts_cam_h = np.vstack((pts_cam, np.ones((1,pts_cam.shape[1]))))
    pts_img = P2 @ pts_cam_h
    pts_img /= pts_img[2]

    u = pts_img[0]
    v = pts_img[1]

    mask_img = (u>=0)&(u<w)&(v>=0)&(v<h)

    u = u[mask_img].astype(int)
    v = v[mask_img].astype(int)
    z_img = z[mask_img]

    # ---- YOLO detection (strict filter) ----
    result = model(img, verbose=False)[0]

    detections = []

    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls not in ALLOWED:
            continue
        if conf < 0.75:
            continue

        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        detections.append([int(x1),int(y1),int(x2),int(y2),ALLOWED[cls]])

    # ---- tracking ----
    tracks = tracker.update(detections)

    # ---- draw + depth ----
    for tid,(x1,y1,x2,y2),label in tracks:

        mask = (u>=x1)&(u<=x2)&(v>=y1)&(v<=y2)
        depths = z_img[mask]
        depths = depths[(depths>1)&(depths<80)]

        dist = np.median(depths) if len(depths) else None

        text = f"ID {tid} | {label}"
        if dist is not None:
            text += f" | {dist:.1f} m"

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img,text,(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2)

    out.write(img)
    del img


out.release()
cv2.destroyAllWindows()
print("Saved:", OUTPUT_VIDEO)
