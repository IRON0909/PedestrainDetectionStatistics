# PedestrianDetection_FastReID.py
'''
import sys,os,cv2

import torch

os.chdir("D:/PedestrainDetection_Packet_Test/yolo_fastreid")
sys.path.append("D:/PedestrainDetection_Packet_Test/yolo_fastreid/fast-reid")

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor


# import torch
# print("PyTorch 版本:", torch.__version__)
# print("GPU 可用:", torch.cuda.is_available())
#
# import sys
# sys.path.append("D:/PedestrainDetection_Packet_Test/Fastreid_test/fast-reid")
# import fastreid
# print("FastReID 路径:", fastreid.__file__)



#加载配置
cfg = get_cfg()
cfg.merge_from_file("D:/PedestrainDetection_Packet_Test/yolo_fastreid/fast-reid/configs/Market1501/bagtricks_R50.yml")
print("加载的 DEPTH:", cfg.MODEL.BACKBONE.DEPTH)
cfg.MODEL.WEIGHTS = "D:/PedestrainDetection_Packet_Test/yolo_fastreid/output/model_best.pth"
cfg.MODEL.DEVICE = "cuda"

predictor = DefaultPredictor(cfg)

# 读取并预处理图像
img = cv2.imread("D:/PedestrainDetection_Packet_Test/yolo_fastreid/img/General.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 转为 Tensor: [H, W, 3] -> [3, H, W] -> [1, 3, H, W]
tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
tensor = tensor / 255.0  # 归一化


# 调用 FastReID
outputs = predictor(tensor)
features = outputs["features"]
feat = features[0].cpu().numpy()  # shape: (512,)
# features=features.cpu().numpy()

print("FastReID 特征提取成功，shape:", features.shape)
'''

# Pedestrian_Feature_Extraction.py

import sys
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO


# 配置 FastReID 路径
fastreid_path = "/fast-reid"
sys.path.insert(0, fastreid_path)

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor


# GPU / CPU 切换

USE_CUDA = True
device = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# 配置 YOLOv8
yolo_model_path = "../Img_test/yolov8s.pt"  # 可替换成自训练权重
yolo_model = YOLO(yolo_model_path)


# 配置 FastReID

cfg_path = os.path.join(fastreid_path, "configs/Market1501/bagtricks_R50.yml")
weights_path = "/output/model_best.pth"  # 训练好的权重

cfg = get_cfg()
cfg.merge_from_file(cfg_path)
cfg.MODEL.DEVICE = device
cfg.MODEL.WEIGHTS = weights_path
cfg.MODEL.HEADS.NAME = "EmbeddingHead"  # 保证和 checkpoint 匹配
cfg.MODEL.HEADS.EMBEDDING_DIM = 512    # 和训练一致

predictor = DefaultPredictor(cfg)


# 图像读取

img_path = "/img/FrenchWeaver.jpg"
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# YOLO 行人检测

results = yolo_model(img_bgr)
person_boxes = []

for box in results[0].boxes:  # YOLOv8 输出
    if int(box.cls[0]) == 0:  # 只取行人类别
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_boxes.append((x1, y1, x2, y2))

print(f"检测到 {len(person_boxes)} 个行人")


# FastReID 特征提取
for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
    crop = img_rgb[y1:y2, x1:x2]

    # 转 tensor: [H, W, 3] -> [1, 3, H, W]
    tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = tensor.to(device)

    # FastReID 特征提取
    with torch.no_grad():
        features = predictor(tensor)  # predictor 返回的就是特征 tensor
        # shape: [1, 512]

    # 去掉 batch 维度
    feat = features.squeeze(0).cpu().numpy()  # shape: (512,)
    print(f"Person {idx+1} 特征向量 shape: {feat.shape}")

    # 可视化
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

scale_percent =200  # 缩放百分比，例如 50%
width = int(img_bgr.shape[1] * scale_percent / 100)
height = int(img_bgr.shape[0] * scale_percent / 100)
dim = (width, height)

resized_img = cv2.resize(img_bgr, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Detections", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()