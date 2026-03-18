import sys, os, cv2, time
import torch
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# FastReID 路径
# -------------------------------
fastreid_path = "/"
sys.path.insert(0, fastreid_path)

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor

# -------------------------------
# 配置 FastReID
# -------------------------------
cfg_path = "/fast-reid/configs/Market1501/bagtricks_R50.yml"
weights_path = "/output/model_best.pth"

cfg = get_cfg()
cfg.merge_from_file(cfg_path)
cfg.MODEL.WEIGHTS = weights_path
cfg.MODEL.DEVICE = "cuda"  # GPU
predictor = DefaultPredictor(cfg)

# -------------------------------
# 视频/图像读取
# -------------------------------
video_path = "D:/PedestrainDetection_Packet_Test/yolo_fastreid/test_video.mp4"
cap = cv2.VideoCapture(0)

# 简单跟踪 ID
track_counter = 0
device = torch.device("cuda")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_bgr = frame.copy()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # TODO: 你这里可以换成 YOLO 检测
    # 假设 person_boxes 是检测出的行人框 [[x1,y1,x2,y2],...]
    person_boxes = [
        [50, 60, 150, 300],
        [200, 100, 300, 350],
        [400, 50, 500, 300]
    ]

    features_list = []

    for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        tensor = torch.from_numpy(crop).permute(2,0,1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(device)

        outputs = predictor(tensor)
        features = outputs["features"].cpu().numpy().flatten()  # shape: (512,)
        features_list.append(features)

        # 可视化边框
        cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img_bgr, f"ID:{idx+1}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # -------------------------------
    # 特征相似度计算
    # -------------------------------
    if len(features_list) >= 2:
        feats = np.stack(features_list)  # shape: (N,512)
        # 余弦相似度矩阵
        sim_matrix = np.dot(feats, feats.T)
        norms = np.linalg.norm(feats, axis=1)
        sim_matrix = sim_matrix / norms[:, None] / norms[None, :]

        # 在图像上显示两人相似度
        for i in range(len(person_boxes)):
            for j in range(i+1, len(person_boxes)):
                x1i, y1i, _, _ = person_boxes[i]
                x1j, y1j, _, _ = person_boxes[j]
                sim_score = sim_matrix[i,j]
                text = f"{sim_score:.2f}"
                x_mid = (x1i + x1j)//2
                y_mid = min(y1i, y1j) - 10
                cv2.putText(img_bgr, text, (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # 可选：显示热力图
        plt.figure(figsize=(4,4))
        plt.imshow(sim_matrix, cmap='hot', vmin=0, vmax=1)
        plt.colorbar()
        plt.title("Person Feature Similarity")
        plt.show()

    # -------------------------------
    # 显示结果
    # -------------------------------
    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detections", 800, 600)
    cv2.imshow("Detections", img_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
