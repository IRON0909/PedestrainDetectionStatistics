'''
import os
import sys
import traceback

# -------------------------------
# 0. 设置是否使用 GPU
# -------------------------------
USE_CUDA = True  # GPU 训练

# -------------------------------
# 1. 环境变量设置
# -------------------------------
if USE_CUDA:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用 GPU 0
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 禁用 GPU
os.environ["FASTREID_DATASETS"] = "D:/PedestrainDetection_Packet_Test/Fastreid_test/fast-reid/datasets"

# -------------------------------
# 2. 设置工作目录和路径
# -------------------------------
os.chdir("D:/PedestrainDetection_Packet_Test/Fastreid_test")
sys.path.append("D:/PedestrainDetection_Packet_Test/Fastreid_test/fast-reid")

# -------------------------------
# 3. 导入 FastReID
# -------------------------------
import torch
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer

# -------------------------------
# 4. 创建配置
# -------------------------------
cfg = get_cfg()
cfg_file = "D:/PedestrainDetection_Packet_Test/Fastreid_test/fast-reid/configs/Market1501/bagtricks_R50.yml"
cfg.merge_from_file(cfg_file)

# -------------------------------
# 5. GPU 设置
# -------------------------------
cfg.MODEL.DEVICE = "cuda" if USE_CUDA else "cpu"
cfg.SOLVER.AMP.ENABLED = False
cfg.SOLVER.IMS_PER_BATCH = 8 if USE_CUDA else 4      # GPU batch 可以大一点
cfg.SOLVER.BASE_LR = 0.00035                          # GPU 默认学习率
cfg.DATALOADER.NUM_WORKERS = 4                        # GPU 可用多进程
cfg.DATALOADER.PIN_MEMORY = USE_CUDA                  # GPU pin memory
cfg.SOLVER.MAX_ITER = 100                              # 测试用小迭代
cfg.OUTPUT_DIR = "./output/"

# -------------------------------
# 6. 数据集检查
# -------------------------------
dataset_path = "D:/PedestrainDetection_Packet_Test/Fastreid_test/fast-reid/datasets/Market-1501-v15.09.15"
if not os.path.exists(dataset_path):
    print(f"警告: 数据集路径不存在: {dataset_path}")
    print("正在创建测试目录结构...")
    os.makedirs(os.path.join(dataset_path, "bounding_box_train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "bounding_box_test"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "query"), exist_ok=True)
    print("测试目录结构已创建")

cfg.DATASETS.NAMES = ("Market1501",)

# -------------------------------
# 7. 创建训练器并训练
# -------------------------------


try:
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # 关键：将 pixel_mean 和 pixel_std 移动到同设备（GPU）
    device = torch.device(cfg.MODEL.DEVICE)
    trainer.model.pixel_mean = trainer.model.pixel_mean.to(device)
    trainer.model.pixel_std  = trainer.model.pixel_std.to(device)

    print("开始训练...")
    trainer.train()
    print("训练完成！")

except Exception as e:
    print(f"训练失败: {e}")
    traceback.print_exc()
'''

import os
import sys
import traceback

def main():
    # -------------------------------
    # 0. 设置是否使用 GPU
    # -------------------------------
    USE_CUDA = True  # GPU 训练

    # -------------------------------
    # 1. 环境变量设置
    # -------------------------------
    if USE_CUDA:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用 GPU 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 禁用 GPU
    os.environ["FASTREID_DATASETS"] = "D:/PedestrainDetection_Packet_Test/Fastreid_test/fast-reid/datasets"

    # -------------------------------
    # 2. 设置工作目录和路径
    # -------------------------------
    os.chdir("D:/PedestrainDetection_Packet_Test/yolo_fastreid")
    sys.path.append("D:/PedestrainDetection_Packet_Test/yolo_fastreid/fast-reid")

    # -------------------------------
    # 3. 导入 FastReID
    # -------------------------------
    import torch
    from fastreid.config import get_cfg
    from fastreid.engine import DefaultTrainer

    # -------------------------------
    # 4. 创建配置
    # -------------------------------
    cfg = get_cfg()
    cfg_file = "D:/PedestrainDetection_Packet_Test/yolo_fastreid/fast-reid/configs/Market1501/bagtricks_R50.yml"
    cfg.merge_from_file(cfg_file)

    # -------------------------------
    # 5. GPU 设置
    # -------------------------------
    cfg.MODEL.DEVICE = "cuda" if USE_CUDA else "cpu"
    cfg.SOLVER.AMP.ENABLED = False
    cfg.SOLVER.IMS_PER_BATCH = 8 if USE_CUDA else 4      # GPU batch 可以大一点
    cfg.SOLVER.BASE_LR = 0.00035                          # GPU 默认学习率
    cfg.DATALOADER.NUM_WORKERS = 4                        # GPU 可用多进程
    cfg.DATALOADER.PIN_MEMORY = USE_CUDA                  # GPU pin memory
    cfg.SOLVER.MAX_ITER = 100                              # 测试用小迭代
    cfg.OUTPUT_DIR = "./output/"

    # -------------------------------
    # 6. 数据集检查
    # -------------------------------
    dataset_path = "D:/PedestrainDetection_Packet_Test/yolo_fastreid/fast-reid/datasets/Market-1501-v15.09.15"
    if not os.path.exists(dataset_path):
        print(f"警告: 数据集路径不存在: {dataset_path}")
        print("正在创建测试目录结构...")
        os.makedirs(os.path.join(dataset_path, "bounding_box_train"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "bounding_box_test"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "query"), exist_ok=True)
        print("测试目录结构已创建")

    cfg.DATASETS.NAMES = ("Market1501",)

    # -------------------------------
    # 7. 创建训练器并训练
    # -------------------------------
    try:
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)

        # 关键：将 pixel_mean 和 pixel_std 移动到同设备（GPU）
        device = torch.device(cfg.MODEL.DEVICE)
        trainer.model.pixel_mean = trainer.model.pixel_mean.to(device)
        trainer.model.pixel_std  = trainer.model.pixel_std.to(device)

        print("开始训练...")
        trainer.train()
        print("训练完成！")

    except Exception as e:
        print(f"训练失败: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    import torch
    # Windows 下多进程 DataLoader 安全
    torch.multiprocessing.freeze_support()
    main()
