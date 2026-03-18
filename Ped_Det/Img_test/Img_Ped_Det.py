import os
import cv2
import torch
import numpy as np
import sys

# -------------------------------
# 配置 FastReID 路径
# -------------------------------
fastreid_path = "/fast-reid"
sys.path.insert(0, fastreid_path)

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor

# -------------------------------
# 配置 YOLOv8
# -------------------------------
from ultralytics import YOLO
yolo_model = YOLO("yolov8s.pt")  # 可换成自训练模型

# -------------------------------
# 初始化 FastReID
# -------------------------------
cfg_path = os.path.join(fastreid_path, "configs/Market1501/bagtricks_R50.yml")
weights_path = os.path.join(fastreid_path, "D:\PedestrainDetection_Packet_Test\yolo_fastreid\output\model_best.pth")

cfg = get_cfg()
cfg.merge_from_file(cfg_path)
cfg.MODEL.WEIGHTS = weights_path
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)
device = torch.device(cfg.MODEL.DEVICE)

# -------------------------------
# 读取图像
# -------------------------------
img_path = "/img/HK_Ped.png"
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# -------------------------------
# YOLO 检测行人
# -------------------------------
results = yolo_model(img_bgr)[0]
person_boxes = []

for box, cls in zip(results. boxes.xyxy, results.boxes.cls):
    if int(cls) == 0:  # 只处理行人
        x1, y1, x2, y2 = map(int, box)
        person_boxes.append((x1, y1, x2, y2))

print(f"检测到 {len(person_boxes)} 个行人")

# -------------------------------
# FastReID 提取特征
# -------------------------------
features_list = []

for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
    crop = img_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        continue
    tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = tensor.to(device)
    features = predictor(tensor)  # 返回 Tensor: [1, 512]
    features_list.append(features.cpu().flatten().numpy())

features_array = np.array(features_list)  # shape: (num_persons, 512)
num_persons = features_array.shape[0]

# -------------------------------
# 计算余弦相似度矩阵
# -------------------------------
norms = np.linalg.norm(features_array, axis=1, keepdims=True)
features_normed = features_array / norms
sim_matrix = np.dot(features_normed, features_normed.T)

# 可视化检测框 + 相似度

for i, (x1, y1, x2, y2) in enumerate(person_boxes):
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(img_bgr, f"ID {i+1}", (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

# 在图像上显示每对行人的相似度
for i in range(num_persons):
    for j in range(i+1, num_persons):
        x1_i, y1_i, x2_i, y2_i = person_boxes[i]
        x1_j, y1_j, x2_j, y2_j = person_boxes[j]
        # 相似度
        sim_score = sim_matrix[i, j]
        # # 显示在两个人中间
        # cx = (x1_i + x1_j) // 2
        # cy = (y1_i + y1_j) // 2
        # cv2.putText(img_bgr, f"{sim_score:.2f}", (cx, cy),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        print("ID ",i+1," similarity ","ID ",j+1," is " ,sim_score)

# 显示窗口
cv2.namedWindow("Detections", cv2.WINDOW_AUTOSIZE)
# cv2.resizeWindow("Detections", 1280, 720)
cv2.imshow("Detections", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
#模型面对相似衣着的目标区分能力差，其余弦相似度高