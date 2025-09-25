# models/cnn_model.py
import math
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 基本元件 ─────────────────────────────────────────────────────────────────────

class SqueezeExcite(nn.Module):
    def __init__(self, c: int, se_ratio: float = 0.25):
        super().__init__()
        c_se = max(1, int(c * se_ratio))
        self.fc1 = nn.Conv2d(c, c_se, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(c_se, c, kernel_size=1, bias=True)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k, s=1, g=1):
        super().__init__()
        p = (k - 1) // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MBConv(nn.Module):
    """
    EfficientNet 風格的 MBConv (expansion → depthwise → SE → projection)。
    支援 stride {1,2} 與殘差連接。
    """
    def __init__(self, c_in, c_out, k, stride, expand_ratio, se_ratio=0.25, drop_path=0.0):
        super().__init__()
        self.stride = stride
        self.use_res = (stride == 1 and c_in == c_out)
        mid = int(c_in * expand_ratio)

        layers = []
        # expand
        if expand_ratio != 1:
            layers += [ConvBNAct(c_in, mid, 1, 1)]
        else:
            mid = c_in

        # depthwise
        layers += [ConvBNAct(mid, mid, k, stride, g=mid)]
        # SE
        layers += [SqueezeExcite(mid, se_ratio)]
        # project
        layers += [nn.Conv2d(mid, c_out, 1, 1, 0, bias=False),
                   nn.BatchNorm2d(c_out)]
        self.block = nn.Sequential(*layers)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out = x + self.drop_path(out)
        return out


class DropPath(nn.Module):
    # 簡易 Stochastic Depth（訓練時才生效）
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep * random_tensor


# ── EfficientNet-B0 相似骨幹（從零初始化） ─────────────────────────────────────────

class EfficientNetB0Like(nn.Module):
    """
    與 EfficientNet-B0 相似的自家實作：
    Stage 組態參考 B0（寬/深度係數 = 1.0），但完全**隨機初始化**，不依賴 timm/預訓練。
    """
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.3, drop_path_rate: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = 'efficientnet_b0_like'
        self.dropout_rate = dropout_rate

        # B0 stage 配置: (exp, k, c_out, repeats, stride)
        cfg = [
            # stem 之後的各 stage
            (1, 3, 16, 1, 1),
            (6, 3, 24, 2, 2),
            (6, 5, 40, 2, 2),
            (6, 3, 80, 3, 2),
            (6, 5,112, 3, 1),
            (6, 5,192, 4, 2),
            (6, 3,320, 1, 1),
        ]

        # Stem
        layers = []
        c = 32
        layers.append(ConvBNAct(3, c, 3, 2))  # 224→112

        # 分配逐層 drop_path
        total_blocks = sum(r for *_, r, _ in [(e,k,c_out,r,s) for e,k,c_out,r,s in cfg])
        dp_idx = 0

        # Stages
        in_ch = c
        for exp, k, c_out, repeats, stride in cfg:
            for i in range(repeats):
                s = stride if i == 0 else 1
                drop_p = drop_path_rate * dp_idx / max(1, total_blocks - 1)
                layers.append(MBConv(in_ch, c_out, k=k, stride=s, expand_ratio=exp, se_ratio=0.25, drop_path=drop_p))
                in_ch = c_out
                dp_idx += 1

        # Head
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            ConvBNAct(in_ch, 1280, 1, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(1280, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        # Kaiming/He 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # SiLU 也可用
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

    # 與原訓練流程相容的資訊輸出
    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'backbone': self.backbone_name,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'feature_dim': 1280,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2)
        }


# ── 工廠：根據設定建立模型 ────────────────────────────────────────────────────────

def create_retina_model(config: Dict[str, Any]) -> nn.Module:
    """
    若 backbone == 'efficientnet_b0_like' → 從零初始化的自定義模型
    否則保留原行為（可選：如你仍想保留舊路線）。
    """
    backbone = config.get('backbone', 'efficientnet_b0_like')
    num_classes = config.get('num_classes', 4)
    dropout_rate = config.get('dropout_rate', 0.3)

    if backbone == 'efficientnet_b0_like':
        return EfficientNetB0Like(num_classes=num_classes, dropout_rate=dropout_rate)
    else:
        # 可選：若你仍想保留舊的遷移學習路線，放回去；不需要則刪除下面區塊
        import timm
        model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
        return model


# 輔助函數：計算模型 FLOPs
def calculate_model_flops(model: torch.nn.Module, input_size: tuple = (3, 224, 224)) -> int:
    """
    計算模型 FLOPs
    Args:
        model: 任意 torch.nn.Module（例如 EfficientNetB0Like）
        input_size: 輸入大小 (C, H, W)
    Returns:
        int: FLOPs 數量
    """
    from thop import profile

    dummy = torch.randn(1, *input_size)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    return int(flops)


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