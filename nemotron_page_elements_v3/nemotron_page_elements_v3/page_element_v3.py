# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class Exp:
    """
    Configuration class for the page element model.

    This class contains all configuration parameters for the YOLOX-based
    page element detection model, including architecture settings, inference
    parameters, and class-specific thresholds.
    """

    def __init__(self) -> None:
        """Initialize the configuration with default parameters."""
        self.name: str = "page-element-v3"
        self.ckpt: str = "weights.pth"
        self.device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

        # YOLOX architecture parameters
        self.act: str = "silu"
        self.depth: float = 1.00
        self.width: float = 1.00
        self.labels: List[str] = [
            "table",
            "chart",
            "title",
            "infographic",
            "text",
            "header_footer",
        ]
        self.num_classes: int = len(self.labels)

        # Inference parameters
        self.size: Tuple[int, int] = (1024, 1024)
        self.min_bbox_size: int = 0
        self.normalize_boxes: bool = True

        # NMS & thresholding. These can be updated
        self.conf_thresh: float = 0.01
        self.iou_thresh: float = 0.5
        self.class_agnostic: bool = True

        self.thresholds_per_class: Dict[str, float] = {
            "table": 0.1,
            "chart": 0.01,
            "infographic": 0.01,
            "title": 0.1,
            "text": 0.1,
            "header_footer": 0.1,
        }

    def get_model(self) -> nn.Module:
        """
        Get the YOLOX model.

        Builds and returns a YOLOX model with the configured architecture.
        Also updates batch normalization parameters for optimal inference.

        Returns:
            nn.Module: The YOLOX model with configured parameters.
        """
        from nemotron_page_elements_v3.yolox.yolox import YOLOX
        from nemotron_page_elements_v3.yolox.yolo_pafpn import YOLOPAFPN
        from nemotron_page_elements_v3.yolox.yolo_head import YOLOXHead

        # Build model
        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels, act=self.act
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels, act=self.act
            )
            self.model = YOLOX(backbone, head)

        # Update batch-norm parameters
        def init_yolo(M: nn.Module) -> None:
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        self.model.apply(init_yolo)

        return self.model
