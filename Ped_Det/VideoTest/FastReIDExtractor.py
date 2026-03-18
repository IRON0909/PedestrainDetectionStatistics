# fastreid_extractor.py
import torch
import torchvision.transforms as T
from fastreid.config import get_cfg
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer
import cv2
import numpy as np

class FastReIDExtractor:
    def __init__(self, config_file="D:/PedestrainDetection_Packet_Test/yolo_fastreid/fast-reid/configs/Market1501/bagtricks_R50.yml",
                 weight_path="D:/PedestrainDetection_Packet_Test/yolo_fastreid/output/model_final.pth",
                 device="cuda"):
        self.cfg = get_cfg()

        self.cfg.merge_from_file(config_file)

        self.cfg.MODEL.BACKBONE.PRETRAIN = False
        self.cfg.MODEL.BACKBONE.PRETRAIN_PATH = ''
        self.device = device
        if not hasattr(self.cfg.INPUT, "PIXEL_MEAN"):
            self.cfg.INPUT.PIXEL_MEAN = [123.675, 116.28, 103.53]
        if not hasattr(self.cfg.INPUT, "PIXEL_STD"):
            self.cfg.INPUT.PIXEL_STD = [58.395, 57.12, 57.375]
        # 构建模型
        self.model = build_model(self.cfg)
        self.model.to(self.device)
        self.model.eval()

        # 加载权重
        Checkpointer(self.model).load(weight_path)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)
        ])

    @torch.no_grad()
    def extract(self, img_bgr):
        """输入BGR numpy图像，输出 L2-normalized 向量"""
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        feat = self.model(tensor)
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.cpu().numpy()[0]