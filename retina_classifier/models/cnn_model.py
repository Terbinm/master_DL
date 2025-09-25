"""
Retina Damage Classification Model
基於 EfficientNet-B0 的輕量級視網膜損傷分類模型
針對 RTX 4090 優化的高效實現
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Optional, Dict, Any


class RetinaClassifier(nn.Module):
    """
    視網膜損傷分類模型
    基於 EfficientNet-B0 的遷移學習實現
    """

    def __init__(
            self,
            num_classes: int = 4,
            backbone: str = 'efficientnet_b0',
            pretrained: bool = True,
            dropout_rate: float = 0.3,
            freeze_backbone: bool = False
    ):
        """
        初始化模型

        Args:
            num_classes: 分類類別數量
            backbone: 骨幹網路名稱
            pretrained: 是否使用預訓練權重
            dropout_rate: Dropout 比例
            freeze_backbone: 是否凍結骨幹網路
        """
        super(RetinaClassifier, self).__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone
        self.dropout_rate = dropout_rate

        # 載入 EfficientNet-B0 骨幹網路
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # 移除原始分類頭
            global_pool=''  # 移除全局池化
        )

        # 獲取特徵維度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]

        # 凍結骨幹網路（可選）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 自定義分類頭
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        # 初始化分類頭權重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化分類頭權重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            x: 輸入圖像 tensor，形狀 [batch_size, 3, 224, 224]

        Returns:
            分類預測結果，形狀 [batch_size, num_classes]
        """
        # 特徵提取
        features = self.backbone(x)

        # 分類預測
        output = self.classifier(features)

        return output

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特徵向量（用於可視化或進一步分析）

        Args:
            x: 輸入圖像 tensor

        Returns:
            特徵向量，形狀 [batch_size, feature_dim]
        """
        features = self.backbone(x)
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
        return pooled_features.flatten(1)

    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """
        解凍骨幹網路進行微調

        Args:
            num_layers: 解凍的層數，None 表示解凍所有層
        """
        backbone_modules = list(self.backbone.children())

        if num_layers is None:
            # 解凍所有層
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # 解凍最後 num_layers 層
            for module in backbone_modules[-num_layers:]:
                for param in module.parameters():
                    param.requires_grad = True

    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'backbone': self.backbone_name,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'feature_dim': self.feature_dim,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2)  # 假設 float32
        }


def create_retina_model(config: Dict[str, Any]) -> RetinaClassifier:
    """
    創建視網膜分類模型的工廠函數

    Args:
        config: 配置字典，包含模型參數

    Returns:
        初始化的模型實例
    """
    model = RetinaClassifier(
        num_classes=config.get('num_classes', 4),
        backbone=config.get('backbone', 'efficientnet_b0'),
        pretrained=config.get('pretrained', True),
        dropout_rate=config.get('dropout_rate', 0.3)
    )

    return model


def load_pretrained_model(checkpoint_path: str, config: Dict[str, Any]) -> RetinaClassifier:
    """
    載入預訓練模型

    Args:
        checkpoint_path: 檢查點文件路徑
        config: 模型配置

    Returns:
        載入權重的模型實例
    """
    model = create_retina_model(config)

    # 載入檢查點
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 處理不同的檢查點格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return model


# 輔助函數：計算模型 FLOPs
def calculate_model_flops(model: RetinaClassifier, input_size: tuple = (3, 224, 224)) -> int:
    """
    計算模型的浮點運算次數 (FLOPs)

    Args:
        model: 模型實例
        input_size: 輸入張量尺寸

    Returns:
        FLOPs 數量
    """
    try:
        from fvcore.nn import FlopCountMode, flop_count

        model.eval()
        inputs = torch.randn(1, *input_size)

        flops_dict, _ = flop_count(model, (inputs,), supported_ops=None)
        total_flops = sum(flops_dict.values())

        return total_flops
    except ImportError:
        print("警告: 無法導入 fvcore，跳過 FLOPs 計算")
        return -1


if __name__ == "__main__":
    # 測試模型創建和基本功能
    from config.config import Config

    print("測試視網膜分類模型...")

    # 創建模型
    model_config = Config.get_model_info()
    model = create_retina_model(model_config)

    # 打印模型信息
    info = model.get_model_info()
    print(f"模型: {info['backbone']}")
    print(f"參數總數: {info['total_params']:,}")
    print(f"可訓練參數: {info['trainable_params']:,}")
    print(f"模型大小: {info['model_size_mb']:.2f} MB")

    # 測試前向傳播
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, 224, 224)  # batch_size=2
        output = model(dummy_input)
        print(f"輸出形狀: {output.shape}")
        print(f"輸出範例: {output[0].tolist()}")

    print("模型測試完成！")