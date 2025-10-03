import torch
import torch.nn as nn
from torchvision import models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TransferCNN(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=False, dropout_rate=0.5):
        super().__init__()

        # 使用 MobileNetV3 Large 當 backbone，載入 ImageNet 預訓練權重
        self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)

        # 修改最後分類層（原本是 Linear(1280, 1000) → 改成 Linear(1280, num_classes)）
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

        # 加入 Dropout（避免 overfitting）
        self.dropout = nn.Dropout(dropout_rate)

        # 決定是否要凍結 backbone
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False  # 凍結卷積層參數

    def forward(self, x):
        x = self.backbone(x)   # MobileNetV3 forward
        x = self.dropout(x)    # dropout 避免過擬合
        return 