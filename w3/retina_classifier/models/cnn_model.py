"""
Classic CNN for Retina Damage Classification (train-from-scratch)
不使用遷移學習，從零開始訓練的經典卷積網路
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class ClassicCNN(nn.Module):
    """
    一個穩健的經典 CNN：
    - 5 個卷積區塊（Conv -> BN -> ReLU -> MaxPool）
    - 後接兩層全連接層
    - 不依賴預訓練、所有權重從零初始化
    """
    def __init__(
        self,
        num_classes: int = 4,
        dropout_rate: float = 0.3,
        in_channels: int = 3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = 'classic_cnn'
        self.dropout_rate = dropout_rate

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)  # 空間尺寸減半
            )

        # 較淺到較深的層級，設計為 224x224 輸入
        self.features = nn.Sequential(
            conv_block(in_channels,   32),   # 224 -> 112
            conv_block(32,            64),   # 112 -> 56
            conv_block(64,           128),   # 56  -> 28
            conv_block(128,          256),   # 28  -> 14
            conv_block(256,          512),   # 14  -> 7
        )

        # 全局平均池化 + MLP 頭
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Kaiming 正態初始化，適合 ReLU
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        pooled = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return pooled  # [batch, 512]

    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'backbone': self.backbone_name,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'feature_dim': 512,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2)  # float32
        }


def create_retina_model(config: Dict[str, Any]) -> ClassicCNN:
    """
    與原本工廠函式同名、相容 main.py 的呼叫方式。
    忽略任何 'pretrained' 參數，全部從零開始。
    """
    return ClassicCNN(
        num_classes=config.get('num_classes', 4),
        dropout_rate=config.get('dropout_rate', 0.3)
    )


def load_pretrained_model(checkpoint_path: str, config: Dict[str, Any]) -> ClassicCNN:
    """
    保留函式名稱以避免其他檔案 import 失效，但這裡不載入任何外部預訓練權重。
    只作為「從檢查點恢復」之用。
    """
    model = create_retina_model(config)
    state = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    return model


# （可選）FLOPs 工具，與原介面一致，方便你保留分析流程
def calculate_model_flops(model: nn.Module, input_size: tuple = (3, 224, 224)) -> int:
    try:
        from fvcore.nn import flop_count
        model.eval()
        inputs = torch.randn(1, *input_size)
        flops_dict, _ = flop_count(model, (inputs,), supported_ops=None)
        return int(sum(flops_dict.values()))
    except Exception:
        return -1
