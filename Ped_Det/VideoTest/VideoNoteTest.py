# # yolo_fastreid_camera
# #从摄像头读取视频帧 -> 用 YOLO 检测人 -> 用 FastReID 提取特征向量 -> 去重并标注 ID -> 保存特征到 CSV。
# from os import close
#
# import cv2
# import numpy as np
# import csv
# import json
# import os
# from datetime import datetime
#
# from sympy.codegen import Print
# from ultralytics import YOLO
# from FastReIDExtractor import FastReIDExtractor
# from scipy.spatial.distance import cosine
#
# # 参数
#
# YOLO_MODEL = "yolov8n.pt"
# CONF_THRESH = 0.8 #阈值
# PERSON_CLASS_ID = 0
# SAVE_CSV = "person_features.csv"
#
#
# # 初始化模型
#
# print("加载 YOLOv8 模型...")
# detector = YOLO(YOLO_MODEL)
#
# print("加载 FastReID 模型...")
# reid_extractor = FastReIDExtractor(
#     config_file="D:/PedestrainDetection_Packet_Test/yolo_fastreid/fast-reid/configs/Market1501/bagtricks_R50.yml",
#     weight_path="D:/PedestrainDetection_Packet_Test/yolo_fastreid/output/model_final.pth"
# )
#
# # 准备视频与数据文件
# VIDEO_PATH = "D:/PedestrainDetection_Packet_Test/yolo_fastreid/Video/crossing-in-hong-kong.mp4"
#
# cap = cv2.VideoCapture(VIDEO_PATH)
# if not cap.isOpened():
#     raise RuntimeError("无法打开视频")
#
# '''
# # 准备摄像头与数据文件
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise RuntimeError("无法打开摄像头")
# '''
# # 若 CSV 不存在则写入表头
# if not os.path.exists(SAVE_CSV):
#     with open(SAVE_CSV, 'a+', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["id", "timestamp", "bbox", "conf", "embedding"])
#
#
# def cosine_similarity(vec1, vec2):
#     vec1 = np.array(vec1)
#     vec2 = np.array(vec2)
#
#     sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
#     return sim
#
# #对比
# def l2_match(emb, recent_embeddings, threshold=0.1):
#     emb = np.array(emb)
#     for pid, vec in recent_embeddings:
#         vec = np.array(vec)
#         dist = np.linalg.norm(emb - vec)
#         if dist <= threshold:
#             return pid, dist
#     return None, None
#
# def is_same_person(vec1, vec2, threshold=0.9999992):
#     """
#     判断两个向量是否属于同一人
#     使用归一化余弦相似度，返回 True/False
#     """
#     vec1 = np.array(vec1)
#     vec2 = np.array(vec2)
#
#     # L2 归一化
#     vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
#     vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
#
#     # 计算余弦相似度
#     cosine_similarity = np.dot(vec1_norm, vec2_norm)
#     print(cosine_similarity)
#     return cosine_similarity >= threshold, cosine_similarity
#
#
#
#
# def reid_similarity(vec1, vec2, threshold=0.75):
#     """
#     计算两个ReID特征向量相似度，并判断是否同一人
#     """
#
#     vec1 = np.array(vec1)
#     vec2 = np.array(vec2)
#
#     # L2归一化
#     vec1 = vec1 / (np.linalg.norm(vec1) + 1e-12)
#     vec2 = vec2 / (np.linalg.norm(vec2) + 1e-12)
#
#     # 余弦相似度
#     sim = np.dot(vec1, vec2)
#
#     # 判断是否同一人
#     same_person = sim >= threshold
#
#     return sim, same_person
#
#
# # 余弦相似度函数
#
# def cosine_sim(a, b):
#     features_array = np.array(a,b)  # shape: (num_persons, 512)
#     num_persons = features_array.shape[0]
#     # -------------------------------
#     # 计算余弦相似度矩阵
#     # -------------------------------
#     norms = np.linalg.norm(features_array, axis=1, keepdims=True)
#     features_normed = features_array / norms
#     sim_matrix = np.dot(features_normed, features_normed.T)
#
#
# # 主循环
#
# id_counter = 0
# recent_embeddings = []
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     results = detector(frame, verbose=False)
#     boxes = results[0].boxes
#     if boxes is None:
#         continue
#
#     for box in boxes:
#         cls = int(box.cls.cpu().numpy())
#         conf = float(box.conf.cpu().numpy())
#         if cls != PERSON_CLASS_ID or conf < CONF_THRESH:
#             continue
#
#         xyxy = box.xyxy.cpu().numpy()[0].astype(int).tolist()
#         x1, y1, x2, y2 = xyxy
#         crop = frame[y1:y2, x1:x2]
#         if crop.size == 0:
#             continue
#
#         # FastReID 特征 把人图输入 FastReID 模型，得到一个向量
#         emb = reid_extractor.extract(crop)
#
#         # 余弦相似度去重，emb当前 vec先前
#         assigned_id = None
#         for pid, vec in recent_embeddings:
#             l2_dist,is_same_person = reid_similarity(emb,vec)
#             print(l2_dist)
#             if is_same_person:
#                 assigned_id = pid
#                 break
#
#         if assigned_id is None:
#             id_counter += 1
#             assigned_id = id_counter
#             recent_embeddings.append((assigned_id, emb))
#             #print(l2_dist)
#             if len(recent_embeddings) > 200:
#                 recent_embeddings.pop(0)
#
#
#
#         # 在画面上显示
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#         cv2.putText(frame, f"ID:{assigned_id}", (x1, y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
#
#         # 写入 CSV
#         with open(SAVE_CSV, 'a', newline='') as f:
#
#             writer = csv.writer(f)
#             writer.writerow([
#                 assigned_id,
#                 datetime.now().isoformat(sep=" "),
#                 json.dumps(xyxy),
#                 f"{conf:.3f}",
#                 json.dumps(emb.tolist())
#             ])
#
#
#     cv2.imshow("YOLO + FastReID", frame)
#     if cv2.waitKey(1) & 0xFF == ord('Q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# print(f"特征保存完毕: {SAVE_CSV}")
#
#

#
# #以上为历史版本
#
#




#仅cos对比 未与前帧去重版
'''
import os
import cv2
import torch
import numpy as np
import csv
import json
from datetime import datetime
from ultralytics import YOLO
from FastReIDExtractor import FastReIDExtractor

# -------------------------------
# 参数
# -------------------------------
YOLO_MODEL = "yolov8n.pt"
CONF_THRESH = 0.8
PERSON_CLASS_ID = 0
SAVE_CSV = "person_features.csv"

# -------------------------------
# 初始化模型
# -------------------------------
print("加载 YOLOv8 模型...")
detector = YOLO(YOLO_MODEL)

print("加载 FastReID 模型...")
reid_extractor = FastReIDExtractor(
    config_file="D:/PedestrainDetection_Packet_Test/yolo_fastreid/fast-reid/configs/Market1501/bagtricks_R50.yml",
    weight_path="D:/PedestrainDetection_Packet_Test/yolo_fastreid/output/model_final.pth"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reid_extractor.model.to(device)
reid_extractor.model.eval()

# -------------------------------
# 摄像头/视频
# -------------------------------
VIDEO_PATH = "D:/PedestrainDetection_Packet_Test/yolo_fastreid/Video/crossing-in-hong-kong.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("无法打开")

# 若 CSV 不存在则写入表头
if not os.path.exists(SAVE_CSV):
    with open(SAVE_CSV, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "timestamp", "bbox", "conf", "embedding"])

# -------------------------------
# 工具函数
# -------------------------------
def extract_embedding(model, crop):
    # BGR → RGB
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    # resize 到 FastReID 输入大小
    crop = cv2.resize(crop, (128, 256))
    tensor = torch.from_numpy(crop).permute(2,0,1).unsqueeze(0).float()
    tensor = tensor.to(device)
    with torch.no_grad():
        features = model(tensor)
    # L2 归一化
    #features = features / (features.norm(p=2, dim=1, keepdim=True) + 1e-12)
    return features.cpu().numpy().flatten()

def cosine_sim_matrix(embeddings):
    """
    输入: list of embeddings
    输出: 对称余弦相似度矩阵
    """
    features_array = np.array(embeddings)
    norms = np.linalg.norm(features_array, axis=1, keepdims=True)
    features_normed = features_array / (norms + 1e-12)
    sim_matrix = np.dot(features_normed, features_normed.T)
    return sim_matrix

# -------------------------------
# 主循环
# -------------------------------
id_counter = 0
recent_embeddings = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector(frame, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        cv2.imshow("YOLO + FastReID", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    current_embeddings = []
    assigned_ids = []

    for box in boxes:
        cls = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        if cls != PERSON_CLASS_ID or conf < CONF_THRESH:
            continue

        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int).tolist()
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        emb = extract_embedding(reid_extractor.model, crop)
        current_embeddings.append(emb)

        # -------------------------------
        # 去重 & 分配 ID
        # -------------------------------
        assigned_id = None
        for pid, vec in recent_embeddings:
            cos_sim = np.dot(emb, vec) / (np.linalg.norm(emb) * np.linalg.norm(vec))
            #l2_dist = np.linalg.norm(emb - vec)
            #cos判断
            if cos_sim > 0.9:
                assigned_id = pid
                break

        if assigned_id is None:
            id_counter += 1
            assigned_id = id_counter
            recent_embeddings.append((assigned_id, emb))
            if len(recent_embeddings) > 200:
                recent_embeddings.pop(0)

        assigned_ids.append(assigned_id)

        # -------------------------------
        # 绘制与保存 CSV
        # -------------------------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID:{assigned_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        with open(SAVE_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                assigned_id,
                datetime.now().isoformat(sep=" "),
                json.dumps([x1, y1, x2, y2]),
                f"{conf:.3f}",
                json.dumps(emb.tolist())
            ])

    # -------------------------------
    # 计算余弦相似度矩阵
    # -------------------------------
    if len(current_embeddings) > 0:
        sim_matrix = cosine_sim_matrix(current_embeddings)
        print(f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} 相似度矩阵:\n", sim_matrix)

    cv2.imshow("YOLO + FastReID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"特征保存完成: {SAVE_CSV}")
'''
#from Ped_Det.ModelTrain.Video_Ped_Det import video_path
#from Ped_Det.VideoTest.utils import iou

'''
#COS 与 L2 共同使用 且与前帧去重版 贪心匹配

import os
import cv2
import torch
import numpy as np
import csv
import json
from datetime import datetime
from ultralytics import YOLO
from FastReIDExtractor import FastReIDExtractor

# 参数
YOLO_MODEL = "yolov8n.pt"
CONF_THRESH = 0.8
PERSON_CLASS_ID = 0
SAVE_CSV = "person_features.csv"

COS_THRESHOLD = 0.85
L2_THRESHOLD = 0.6
MAX_HISTORY = 200  # 保存的历史特征数量


# 初始化模型
print("加载 YOLOv8 模型...")
detector = YOLO(YOLO_MODEL)

print("加载 FastReID 模型...")
reid_extractor = FastReIDExtractor(
    config_file="D:/PedestrainDetection_Packet_Test/yolo_fastreid/fast-reid/configs/Market1501/bagtricks_R50.yml",
    weight_path="D:/PedestrainDetection_Packet_Test/yolo_fastreid/output/model_final.pth"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reid_extractor.model.to(device)
reid_extractor.model.eval()

# 视频路径

VIDEO_PATH = "D:/PedestrainDetection_Packet_Test/yolo_fastreid/Video/crossing-in-hong-kong.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("无法打开视频")

# CSV文件表头
if not os.path.exists(SAVE_CSV):
    with open(SAVE_CSV, 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "timestamp", "bbox", "conf", "embedding"])


# 提取特征
def extract_embedding(model, crop):
    #裁剪图像 ->FastReID 特征向量
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (128, 256))
    tensor = torch.from_numpy(crop).permute(2,0,1).unsqueeze(0).float()
    tensor = tensor.to(device)
    with torch.no_grad():
        features = model(tensor)
    return features.cpu().numpy().flatten()

def is_same_person_dual(new_emb, known_emb, cos_threshold=COS_THRESHOLD, l2_threshold=L2_THRESHOLD):
    #使用余弦相似度 + L2 距离判断是否为同一人
    new_emb = np.array(new_emb)
    known_emb = np.array(known_emb)

    # L2归一化
    new_emb /= (np.linalg.norm(new_emb) + 1e-12)
    known_emb /= (np.linalg.norm(known_emb) + 1e-12)

    cos_sim = np.dot(new_emb, known_emb)
    l2_dist = np.linalg.norm(new_emb - known_emb)
    same_person = (cos_sim >= cos_threshold) and (l2_dist <= l2_threshold)
    return same_person, cos_sim, l2_dist

def cosine_sim_matrix(embeddings):
    #对称余弦相似度矩阵
    features_array = np.array(embeddings)
    norms = np.linalg.norm(features_array, axis=1, keepdims=True)
    features_normed = features_array / (norms + 1e-12)
    sim_matrix = np.dot(features_normed, features_normed.T)
    return sim_matrix


# 主循环
id_counter = 0
all_embeddings = []  # 历史ID + embedding
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector(frame, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        cv2.imshow("YOLO + FastReID", frame)
        if cv2.waitKey(1) & 0xFF == ord('Q'):
            break
        continue

    current_embeddings = []
    assigned_ids = []

    for box in boxes:
        cls = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        if cls != PERSON_CLASS_ID or conf < CONF_THRESH:
            continue

        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int).tolist()
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        emb = extract_embedding(reid_extractor.model, crop)
        current_embeddings.append(emb)

        # 对比历史特征
        assigned_id = None
        for pid, known_emb in all_embeddings:
            same, cos_sim, l2_dist = is_same_person_dual(emb, known_emb)
            if same:
                assigned_id = pid
                print(f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} 匹配历史ID:{pid}, cos={cos_sim:.4f}, L2={l2_dist:.4f}")
                break

        if assigned_id is None:
            id_counter += 1
            assigned_id = id_counter
            all_embeddings.append((assigned_id, emb))
            if len(all_embeddings) > MAX_HISTORY:
                all_embeddings.pop(0)

        assigned_ids.append(assigned_id)

        # 绘制与保存 CSV
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID:{assigned_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        with open(SAVE_CSV, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                assigned_id,
                datetime.now().isoformat(sep=" "),
                json.dumps([x1, y1, x2, y2]),
                f"{conf:.3f}",
                json.dumps(emb.tolist())
            ])

    # -------------------------------
    # 当前帧余弦相似度矩阵
    # -------------------------------
    if len(current_embeddings) > 0:
        sim_matrix = cosine_sim_matrix(current_embeddings)
        print(f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} 相似度矩阵:\n", sim_matrix)

    cv2.imshow("YOLO + FastReID", frame)
    if cv2.waitKey(1) & 0xFF == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"特征保存完成: {SAVE_CSV}")

'''
'''

#COS 与 L2 共同使用 且与前帧去重版 匈牙利算法匹配


import os
import cv2
import torch
import numpy as np
import csv
from ultralytics import YOLO
from FastReIDExtractor import FastReIDExtractor
from scipy.optimize import linear_sum_assignment
from FastReIDExtractor import FastReIDExtractor
from tracker import Track

# 参数
YOLO_MODEL = "yolov8n.pt"

CONF_THRESH = 0.8
PERSON_CLASS_ID = 0

SAVE_CSV = "person_features.csv"

COS_THRESHOLD = 0.85
L2_THRESHOLD = 0.6

SIM_THRESHOLD = 0.85
MAX_LOST = 30


# 初始化模型
print("加载 YOLOv8...")
detector = YOLO(YOLO_MODEL)

print("加载 FastReID...")
reid_extractor = FastReIDExtractor(
    config_file="D:/PedestrainDetection_Packet_Test/yolo_fastreid/fast-reid/configs/Market1501/bagtricks_R50.yml",
    weight_path="D:/PedestrainDetection_Packet_Test/yolo_fastreid/output/model_final.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reid_extractor.model.to(device)
reid_extractor.model.eval()


# 视频
VIDEO_PATH = "D:/PedestrainDetection_Packet_Test/yolo_fastreid/Video/crossing-in-hong-kong.mp4"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("无法打开视频")


# CSV初始化

if not os.path.exists(SAVE_CSV):
    with open(SAVE_CSV,'w+',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id","timestamp","bbox","conf","embedding"])



# 特征提取
def extract_embedding(model,crop):

    crop = cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop,(128,256))

    tensor = torch.from_numpy(crop).permute(2,0,1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        features = model(tensor)

    return features.cpu().numpy().flatten()



# Cos + L2 判断
def is_same_person_dual(new_emb,known_emb):

    new_emb = np.array(new_emb)
    known_emb = np.array(known_emb)

    new_emb /= (np.linalg.norm(new_emb)+1e-12)
    known_emb /= (np.linalg.norm(known_emb)+1e-12)

    cos_sim = np.dot(new_emb,known_emb)
    l2_dist = np.linalg.norm(new_emb-known_emb)

    same = (cos_sim>=COS_THRESHOLD) and (l2_dist<=L2_THRESHOLD)

    return same,cos_sim,l2_dist


# cost matrix
def compute_cost(tracks,det_features):

    track_features = np.array([t["feature"] for t in tracks])

    track_features /= (np.linalg.norm(track_features,axis=1,keepdims=True)+1e-12)
    det_features /= (np.linalg.norm(det_features,axis=1,keepdims=True)+1e-12)

    sim_matrix = np.dot(track_features,det_features.T)

    cost_matrix = 1-sim_matrix

    return cost_matrix


# 轨迹
tracks = []
next_id = 1



# 主循环
while True:

    ret,frame = cap.read()

    if not ret:
        break


    results = detector(frame,verbose=False)

    boxes = results[0].boxes

    detections=[]
    features=[]
    confs=[]



    # 检测循环
    if boxes is not None:

        for box in boxes:

            cls=int(box.cls.cpu().numpy()[0])
            conf=float(box.conf.cpu().numpy()[0])
            if cls!=PERSON_CLASS_ID or conf<CONF_THRESH:
                continue

            x1,y1,x2,y2=box.xyxy.cpu().numpy()[0].astype(int)

            crop=frame[y1:y2,x1:x2]

            if crop.size==0:
                continue

            emb=extract_embedding(reid_extractor.model,crop)

            detections.append([x1,y1,x2,y2])
            features.append(emb)
            confs.append(conf)


    features=np.array(features)


    # 初始化轨迹
    if len(tracks)==0:

        for i,box in enumerate(detections):

            tracks.append({
                "id":next_id,
                "bbox":box,
                "feature":features[i],
                "lost":0
            })

            next_id+=1


    # 匈牙利匹配

    else:

        if len(features)==0:
            continue

        cost_matrix=compute_cost(tracks,features)

        row,col=linear_sum_assignment(cost_matrix)

        matched_track=set()
        matched_det=set()



        # 更新匹配轨迹
        for r,c in zip(row,col):

            sim=1-cost_matrix[r,c]

            if sim>SIM_THRESHOLD:

                tracks[r]["bbox"]=detections[c]
                tracks[r]["feature"]=features[c]
                tracks[r]["lost"]=0

                matched_track.add(r)
                matched_det.add(c)


        # 未匹配轨迹

        for i,t in enumerate(tracks):

            if i not in matched_track:

                t["lost"]+=1


        # 删除丢失轨迹
        tracks=[t for t in tracks if t["lost"]<MAX_LOST]



        # 新目标
        for i,box in enumerate(detections):

            if i not in matched_det:

                tracks.append({
                    "id":next_id,
                    "bbox":box,
                    "feature":features[i],
                    "lost":0
                })

                next_id+=1


    # 绘制
    for t in tracks:

        x1,y1,x2,y2=map(int,t["bbox"])

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(
            frame,
            f"ID:{t['id']}",
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )


    cv2.imshow("YOLO + FastReID Tracking",frame)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

print("运行结束")

'''
'''
#COS 与 L2 共同使用 且与前帧去重版 匈牙利算法匹配  Kalman预测

import os
import cv2
import torch
import numpy as np
import csv
from datetime import datetime
from ultralytics import YOLO
from Ped_Det.VideoTest.utils import iou
from FastReIDExtractor import FastReIDExtractor
from scipy.optimize import linear_sum_assignment
from tracker import Track

#参数
YOLO_MODEL = "yolov8n.pt"
CONF_THRESH = 0.8
PERSON_CLASS_ID = 0
SIM_THRESHOLD = 0.85
MAX_LOST = 30
MAX_GALLERY = 100
SAVE_CSV = "person_features.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化
detector = YOLO(YOLO_MODEL)

reid_extractor = FastReIDExtractor(
    config_file="D:/PedestrainDetection_Packet_Test/yolo_fastreid/fast-reid/configs/Market1501/bagtricks_R50.yml",
    weight_path="D:/PedestrainDetection_Packet_Test/yolo_fastreid/output/model_best.pth"
)
reid_extractor.model.to(device).eval()
video_path="D:/PedestrainDetection_Packet_Test/yolo_fastreid\Video/crossing-in-hong-kong.mp4"
cap = cv2.VideoCapture(0)

csv_buffer = []

gallery = []
gallery_ids = []

#function of machine

def extract_embeddings_batch(model, crops):
    if len(crops) == 0:
        return np.array([])
    imgs = [cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (128,256)) for c in crops]
    tensor = torch.from_numpy(np.array(imgs)).permute(0,3,1,2).float().to(device)
    with torch.no_grad():
        feats = model(tensor).cpu().numpy()
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    return feats

def find_in_gallery(feature, threshold=0.85):
    if len(gallery) == 0:
        return -1
    sims = np.dot(np.array(gallery), feature)
    idx = np.argmax(sims)
    return gallery_ids[idx] if sims[idx] > threshold else -1

def compute_cost(tracks, detections, features):
    cost = np.zeros((len(tracks), len(detections)))
    for i,t in enumerate(tracks):
        for j,d in enumerate(detections):
            sim = 0.7*np.dot(t.feature, features[j]) + 0.3*iou(t.bbox, d)
            cost[i,j] = 1 - sim
    return cost


#主循环
tracks = []
next_id = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640,360))

    results = detector(frame, verbose=False)
    boxes = results[0].boxes

    detections, crops = [], []

    #检测
    if boxes is not None:
        for box in boxes:
            cls = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            if cls != PERSON_CLASS_ID or conf < CONF_THRESH:
                continue
            x1,y1,x2,y2 = box.xyxy.cpu().numpy()[0].astype(int)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            detections.append([x1,y1,x2,y2])
            crops.append(crop)

    features = extract_embeddings_batch(reid_extractor.model, crops)

    #预测
    for t in tracks:
        t.predict()

    #关联
    if len(tracks) > 0 and len(features) > 0:
        cost = compute_cost(tracks, detections, features)
        row, col = linear_sum_assignment(cost)

        matched_t, matched_d = set(), set()

        for r,c in zip(row,col):
            if 1 - cost[r,c] > SIM_THRESHOLD:
                tracks[r].update(detections[c], features[c])
                matched_t.add(r)
                matched_d.add(c)

        for i,t in enumerate(tracks):
            if i not in matched_t:
                t.lost += 1

    else:
        for t in tracks:
            t.lost += 1

    # 新目标
    for i, box in enumerate(detections):
        if len(features)==0:
            continue

        duplicate = False
        for t in tracks:
            if np.dot(t.feature, features[i]) > 0.9:
                duplicate = True
                break

        if duplicate:
            continue

        matched_id = find_in_gallery(features[i])
        if matched_id != -1:
            tracks.append(Track(matched_id, box, features[i]))
        else:
            tracks.append(Track(next_id, box, features[i]))
            next_id += 1

    #删除轨迹  存入gallery
    new_tracks = []
    for t in tracks:
        if t.lost < MAX_LOST:
            new_tracks.append(t)
        else:
            gallery.append(t.feature)
            gallery_ids.append(t.id)
            if len(gallery) > MAX_GALLERY:
                gallery.pop(0)
                gallery_ids.pop(0)

    tracks = Track.deduplicate_tracks(new_tracks)

    #可视化  CSV
    for t in tracks:
        x1,y1,x2,y2 = map(int, t.bbox)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"ID:{t.id}",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        csv_buffer.append([t.id, datetime.now().isoformat(), t.bbox, "", t.feature.tolist()])

    if len(csv_buffer) > 50:
        with open(SAVE_CSV,'a+',newline='') as f:
            csv.writer(f).writerows(csv_buffer)
        csv_buffer.clear()

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

'''




#+人数统计
import os
import cv2
import torch
import numpy as np
import csv
from datetime import datetime
from ultralytics import YOLO
from Ped_Det.VideoTest.utils import iou
from FastReIDExtractor import FastReIDExtractor
from scipy.optimize import linear_sum_assignment
from tracker import Track

#参数
YOLO_MODEL = "yolov8n.pt"
CONF_THRESH = 0.8
PERSON_CLASS_ID = 0
SIM_THRESHOLD = 0.85
MAX_LOST = 30
MAX_GALLERY = 100
SAVE_CSV = "person_features.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#初始化
detector = YOLO(YOLO_MODEL)

reid_extractor = FastReIDExtractor(
    config_file="D:/PedestrainDetection_Packet_Test/yolo_fastreid/fast-reid/configs/Market1501/bagtricks_R50.yml",
    weight_path="D:/PedestrainDetection_Packet_Test/yolo_fastreid/output/model_best.pth"
)
reid_extractor.model.to(device).eval()
Video_Path="D:/PedestrainDetection_Packet_Test/yolo_fastreid/Video/crossing-in-hong-kong.mp4"
cap = cv2.VideoCapture(Video_Path)

csv_buffer = []

#ReID
gallery = []
gallery_ids = []

#人数统计
total_unique_ids = set()

#function of tools
def extract_embeddings_batch(model, crops):
    if len(crops) == 0:
        return np.array([])
    imgs = [cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (128, 256)) for c in crops]
    tensor = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
    with torch.no_grad():
        feats = model(tensor).cpu().numpy()
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    return feats

def find_in_gallery(feature, threshold=0.85):
    if len(gallery) == 0:
        return -1
    sims = np.dot(np.array(gallery), feature)
    idx = np.argmax(sims)
    return gallery_ids[idx] if sims[idx] > threshold else -1

def compute_cost(tracks, detections, features):
    cost = np.zeros((len(tracks), len(detections)))
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            sim = 0.7 * np.dot(t.feature, features[j]) + 0.3 * iou(t.bbox, d)
            cost[i, j] = 1 - sim
    return cost

#主循环
tracks = []
next_id = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    #检测
    results = detector(frame, verbose=False)
    boxes = results[0].boxes

    detections, crops = [], []

    if boxes is not None:
        for box in boxes:
            cls = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            if cls != PERSON_CLASS_ID or conf < CONF_THRESH:
                continue

            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            detections.append([x1, y1, x2, y2])
            crops.append(crop)

    # ReID特征
    features = extract_embeddings_batch(reid_extractor.model, crops)

    #预测
    for t in tracks:
        t.predict()

    #关联
    if len(tracks) > 0 and len(features) > 0:
        cost = compute_cost(tracks, detections, features)
        row, col = linear_sum_assignment(cost)

        matched_t, matched_d = set(), set()

        for r, c in zip(row, col):
            if 1 - cost[r, c] > SIM_THRESHOLD:
                tracks[r].update(detections[c], features[c])
                matched_t.add(r)
                matched_d.add(c)

        for i, t in enumerate(tracks):
            if i not in matched_t:
                t.lost += 1
    else:
        for t in tracks:
            t.lost += 1

    #新目标
    for i, box in enumerate(detections):
        if len(features) == 0:
            continue

        # 防重复
        duplicate = False
        for t in tracks:
            if np.dot(t.feature, features[i]) > 0.9:
                duplicate = True
                break

        if duplicate:
            continue

        # 查gallery
        matched_id = find_in_gallery(features[i])

        if matched_id != -1:
            tracks.append(Track(matched_id, box, features[i]))
        else:
            tracks.append(Track(next_id, box, features[i]))
            next_id += 1

    #删除轨迹
    new_tracks = []
    for t in tracks:
        if t.lost < MAX_LOST:
            new_tracks.append(t)
        else:
            gallery.append(t.feature)
            gallery_ids.append(t.id)

            if len(gallery) > MAX_GALLERY:
                gallery.pop(0)
                gallery_ids.pop(0)

    tracks = Track.deduplicate_tracks(new_tracks)

    #人数统计
    current_count = len(tracks)

    for t in tracks:
        total_unique_ids.add(t.id)

    total_count = len(total_unique_ids)

    #可视化
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{t.id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        csv_buffer.append([
            t.id,
            datetime.now().isoformat(),
            t.bbox,
            "",
            t.feature.tolist()
        ])

    #显示
    cv2.rectangle(frame, (10, 10), (220, 80), (0, 0, 0), -1)

    cv2.putText(frame, f"Current: {current_count}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"Total: {total_count}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if len(csv_buffer) > 50:
        with open(SAVE_CSV, 'a+', newline='') as f:
            csv.writer(f).writerows(csv_buffer)
        csv_buffer.clear()

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
