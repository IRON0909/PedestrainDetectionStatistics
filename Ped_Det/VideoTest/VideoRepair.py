import cv2
import torch
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


#加载YOLO
detector = YOLO("yolov8n.pt")


#2 加载FastReID模型

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor

cfg = get_cfg()
cfg.merge_from_file("D:/PedestrainDetection_Packet_Test/yolo_fastreid/fast-reid/configs/Market1501/bagtricks_R50.yml")
cfg.MODEL.WEIGHTS = "D:\PedestrainDetection_Packet_Test\yolo_fastreid\output\model_best.pth"
cfg.MODEL.DEVICE = "cuda"

reid_model = DefaultPredictor(cfg)


#跟踪器参数

tracks = []
next_id = 0

MAX_LOST = 30
SIM_THRESHOLD = 0.7

#提取ReID特征

def extract_feature(img):

    img = cv2.resize(img,(128,256))
    img = img[:,:,::-1].astype("float32")/255.0

    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)

    with torch.no_grad():
        feat = reid_model(img)

    feat = feat.squeeze().cpu().numpy()

    return feat


#计算匹配cost矩阵
def compute_cost(tracks, det_features):
    #(512,128)
    track_features = np.array([t["feature"] for t in tracks])
    sim_matrix = cosine_similarity(track_features, det_features)
    cost_matrix = 1 - sim_matrix

    return cost_matrix

# 视频读取
VideoPath="D:\PedestrainDetection_Packet_Test\yolo_fastreid\Video\crossing-in-hong-kong.mp4"
cap = cv2.VideoCapture(VideoPath)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = detector(frame)[0]

    detections = []
    features = []

    # YOLO检测
    for box in results.boxes:

        cls = int(box.cls[0])

        if cls != 0:   # 只检测人
            continue

        x1,y1,x2,y2 = map(int,box.xyxy[0])

        crop = frame[y1:y2,x1:x2]

        if crop.size == 0:
            continue

        feat = extract_feature(crop)

        detections.append([x1,y1,x2,y2])
        features.append(feat)

    features = np.array(features)  #list ->numpy


    # 初始化轨迹
    if len(tracks) == 0:

        for i,box in enumerate(detections):

            tracks.append({
                "id":next_id,
                "bbox":box,
                "feature":features[i],
                "lost":0
            })

            next_id+=1

    else:
        if len(features)==0:
            continue

        cost_matrix = compute_cost(tracks,features)
        row,col = linear_sum_assignment(cost_matrix)
        matched_track=set()
        matched_det=set()


        # 更新匹配轨迹
        for r,c in zip(row,col):

            sim = 1-cost_matrix[r,c]

            if sim>SIM_THRESHOLD:

                tracks[r]["bbox"]=detections[c]
                tracks[r]["feature"]=features[c]
                tracks[r]["lost"]=0

                matched_track.add(r)
                matched_det.add(c)


        #  更新丢失轨迹

        for i,t in enumerate(tracks):

            if i not in matched_track:
                t["lost"]+=1

        tracks=[t for t in tracks if t["lost"]<MAX_LOST]

        # 创建新轨迹
        for i in range(len(detections)):
            if i not in matched_det:
                tracks.append({
                    "id":next_id,
                    "bbox":detections[i],
                    "feature":features[i],
                    "lost":0
                })

                next_id+=1

    #绘制
    for t in tracks:
        x1,y1,x2,y2=t["bbox"]

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,
                    f"ID {t['id']}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

    cv2.imshow("tracking",frame)
#
    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()