"""
Retina Damage Classification - Configuration Management
針對 RTX 4090 + PyTorch 2.5.1 + CUDA 12.1 優化配置
"""

import torch
import os
from pathlib import Path


class Config:
    """統一配置管理類別"""

    # ==================== 硬體配置 ====================
    # 基於 RTX 4090 的最佳化設定
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 8  # RTX 4090 建議使用 8-12 個工作進程
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True

    # 混合精度訓練 (RTX 4090 支援 Tensor Cores)
    USE_MIXED_PRECISION = True

    # ==================== 路徑配置 ====================
    # 專案根目錄
    PROJECT_ROOT = Path(__file__).parent.parent

    # 數據路徑
    DATA_ROOT = PROJECT_ROOT / 'data'
    RAW_DATA_PATH = DATA_ROOT / 'raw'
    PROCESSED_DATA_PATH = DATA_ROOT / 'processed'

    # 模型和日誌路徑
    CHECKPOINTS_PATH = PROJECT_ROOT / 'checkpoints'
    LOGS_PATH = PROJECT_ROOT / 'logs'

    # 創建必要目錄
    for path in [DATA_ROOT, RAW_DATA_PATH, PROCESSED_DATA_PATH,
                 CHECKPOINTS_PATH, LOGS_PATH]:
        path.mkdir(parents=True, exist_ok=True)

    # ==================== 數據配置 ====================
    # 視網膜疾病分類類別
    CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    NUM_CLASSES = len(CLASS_NAMES)

    # 圖像處理參數
    IMAGE_SIZE = (224, 224)  # EfficientNet-B0 標準輸入尺寸
    MEAN = [0.485, 0.456, 0.406]  # ImageNet 標準化參數
    STD = [0.229, 0.224, 0.225]

    # ==================== 模型配置 ====================
    BACKBONE = 'classic_cnn'
    PRETRAINED = False
    DROPOUT_RATE = 0.3

    # ==================== 訓練配置 ====================
    # RTX 4090 優化的批次大小 (24GB VRAM)
    BATCH_SIZE = 64  # 可根據需要調整至 128
    VALIDATION_BATCH_SIZE = 64

    # 訓練週期
    NUM_EPOCHS = 50
    PATIENCE = 10  # 早停耐心值

    # 優化器配置
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # 學習率調度
    LR_SCHEDULE = 'cosine'  # cosine, step, plateau
    LR_GAMMA = 0.1  # step 調度的衰減係數
    LR_STEP_SIZE = 10  # step 調度的步長

    # ==================== 數據增強配置 ====================
    # 訓練時數據增強
    TRAIN_TRANSFORMS = {
        'rotation_range': 15,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'horizontal_flip': True,
        'zoom_range': 0.1,
        'brightness_range': [0.8, 1.2]
    }

    # ==================== TensorBoard 配置 ====================
    LOG_INTERVAL = 10  # 每隔幾個 batch 記錄一次
    SAVE_MODEL_INTERVAL = 5  # 每隔幾個 epoch 保存一次模型

    # ==================== 系統配置 ====================
    # 隨機種子設定
    RANDOM_SEED = 42

    # 模型檢查點
    CHECKPOINT_FORMAT = 'retina_classifier_epoch_{epoch:02d}_acc_{val_acc:.4f}.pth'
    BEST_MODEL_PATH = CHECKPOINTS_PATH / 'best_model.pth'

    # 結果輸出
    RESULTS_PATH = PROJECT_ROOT / 'results'
    RESULTS_PATH.mkdir(exist_ok=True)

    @classmethod
    def print_config(cls):
        """打印當前配置信息"""
        print("=" * 60)
        print("視網膜損傷分類系統 - 配置信息")
        print("=" * 60)
        print(f"設備: {cls.DEVICE}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"顯存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"批次大小: {cls.BATCH_SIZE}")
        print(f"圖像尺寸: {cls.IMAGE_SIZE}")
        print(f"分類數量: {cls.NUM_CLASSES}")
        print(f"分類名稱: {cls.CLASS_NAMES}")
        print(f"混合精度: {'啟用' if cls.USE_MIXED_PRECISION else '停用'}")
        print("=" * 60)

    @classmethod
    def get_model_info(cls):
        """返回模型相關配置"""
        return {
            'backbone': cls.BACKBONE,
            'num_classes': cls.NUM_CLASSES,
            'image_size': cls.IMAGE_SIZE,
            'dropout_rate': cls.DROPOUT_RATE,
            'pretrained': cls.PRETRAINED
        }

    @classmethod
    def get_training_info(cls):
        """返回訓練相關配置"""
        return {
            'batch_size': cls.BATCH_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'num_epochs': cls.NUM_EPOCHS,
            'weight_decay': cls.WEIGHT_DECAY,
            'patience': cls.PATIENCE,
            'lr_schedule': cls.LR_SCHEDULE
        }