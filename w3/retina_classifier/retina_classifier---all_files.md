# retina_classifier---all_files.md
# 目錄結構
```
./
├── main.py
├── README.md
├── requirements.txt
├── retina_classifier---all_files.md
├── __init__.py
├── config/
│   ├── config.py
│   ├── __init__.py
├── models/
│   ├── cnn_model.py
│   ├── utils.py
│   ├── __init__.py
├── utils/
│   ├── data_loader.py
│   ├── trainer.py
│   ├── visualizer.py
│   ├── __init__.py
```
---

main.py:
```
"""
Retina Damage Classification - Main Training Script
視網膜損傷分類系統主程式
針對 RTX 4090 + PyTorch 2.5.1 + CUDA 12.1 優化
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import random
import warnings
from typing import Dict, Any

# 本地模組導入
from config.config import Config
from models.cnn_model import create_retina_model
from utils.data_loader import create_data_loaders, calculate_class_weights
from utils.trainer import RetinaTrainer
from utils.visualizer import create_tensorboard_visualizer

warnings.filterwarnings('ignore')


def set_random_seeds(seed: int = 42):
    """設定隨機種子以確保結果可重現"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"隨機種子設定為: {seed}")


def check_environment():
    """檢查運行環境"""
    print("=" * 60)
    print("環境檢查")
    print("=" * 60)

    # PyTorch 版本
    print(f"PyTorch 版本: {torch.__version__}")

    # CUDA 可用性
    if torch.cuda.is_available():
        print(f"CUDA 可用: True")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 數量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        print(f"當前 GPU: {torch.cuda.current_device()}")
    else:
        print("CUDA 可用: False")
        print("警告: 將使用 CPU 進行訓練，速度會很慢！")

    print("=" * 60)


def check_data_structure():
    """檢查數據結構"""
    print("\n數據結構檢查")
    print("-" * 40)

    data_root = Config.RAW_DATA_PATH
    print(f"數據根目錄: {data_root}")

    if not data_root.exists():
        print(f"❌ 數據目錄不存在: {data_root}")
        print("\n請按以下結構準備數據:")
        print("data/raw/")
        print("├── train/")
        print("│   ├── CNV/")
        print("│   ├── DME/")
        print("│   ├── DRUSEN/")
        print("│   └── NORMAL/")
        print("├── val/")
        print("│   ├── CNV/")
        print("│   ├── DME/")
        print("│   ├── DRUSEN/")
        print("│   └── NORMAL/")
        print("└── test/ (可選)")
        print("    ├── CNV/")
        print("    ├── DME/")
        print("    ├── DRUSEN/")
        print("    └── NORMAL/")
        return False

    # 檢查各個分割
    splits = ['train', 'val', 'test']
    available_splits = []

    for split in splits:
        split_dir = data_root / split
        if split_dir.exists():
            available_splits.append(split)
            print(f"✓ {split.upper()} 目錄存在")

            # 檢查類別目錄
            for class_name in Config.CLASS_NAMES:
                class_dir = split_dir / class_name
                if class_dir.exists():
                    # 計算圖像數量
                    image_count = len(list(class_dir.glob('*.jpg'))) + \
                                  len(list(class_dir.glob('*.jpeg'))) + \
                                  len(list(class_dir.glob('*.png')))
                    print(f"  - {class_name}: {image_count} 圖像")
                else:
                    print(f"  ❌ {class_name} 目錄不存在")
        else:
            print(f"- {split.upper()} 目錄不存在")

    if not available_splits:
        print("❌ 未找到任何數據分割")
        return False

    if 'train' not in available_splits:
        print("❌ 缺少訓練數據")
        return False

    print(f"✓ 數據結構檢查通過，可用分割: {available_splits}")
    return True


def train_model():
    """主要的模型訓練函數"""
    print("\n" + "=" * 60)
    print("開始訓練視網膜損傷分類模型")
    print("=" * 60)

    # 檢查環境和數據
    check_environment()

    if not check_data_structure():
        print("❌ 數據結構檢查失敗，請修正後重試")
        return

    # 設定隨機種子
    set_random_seeds(Config.RANDOM_SEED)

    # 打印配置信息
    Config.print_config()

    try:
        # 1. 創建數據載入器
        print("\n步驟 1: 創建數據載入器...")
        data_loaders = create_data_loaders(
            data_dir=str(Config.RAW_DATA_PATH),
            class_names=Config.CLASS_NAMES,
            batch_size=Config.BATCH_SIZE,
            val_batch_size=Config.VALIDATION_BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            persistent_workers=Config.PERSISTENT_WORKERS,
            image_size=Config.IMAGE_SIZE,
            transform_config=Config.TRAIN_TRANSFORMS
        )

        train_loader = data_loaders['train']
        val_loader = data_loaders.get('val')
        test_loader = data_loaders.get('test')

        # 2. 計算類別權重（處理不平衡數據）
        print("\n步驟 2: 計算類別權重...")
        class_weights = calculate_class_weights(train_loader.dataset)
        print(f"類別權重: {class_weights.tolist()}")

        # 3. 創建模型
        print("\n步驟 3: 創建模型...")
        model_config = Config.get_model_info()
        model = create_retina_model(model_config)

        # 打印模型信息
        model_info = model.get_model_info()
        print(f"模型架構: {model_info['backbone']}")
        print(f"參數總數: {model_info['total_params']:,}")
        print(f"可訓練參數: {model_info['trainable_params']:,}")
        print(f"模型大小: {model_info['model_size_mb']:.2f} MB")

        # 4. 創建訓練器
        print("\n步驟 4: 創建訓練器...")
        trainer = RetinaTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=Config.DEVICE,
            use_mixed_precision=Config.USE_MIXED_PRECISION,
            checkpoints_dir=str(Config.CHECKPOINTS_PATH),
            class_names=Config.CLASS_NAMES
        )

        # 5. 設置優化器和調度器
        print("\n步驟 5: 設置優化器和調度器...")
        training_config = Config.get_training_info()
        trainer.setup_optimizer_and_scheduler(
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            scheduler_type=training_config['lr_schedule'],
            T_max=training_config['num_epochs'],
            step_size=Config.LR_STEP_SIZE,
            gamma=Config.LR_GAMMA
        )

        # 6. 設置損失函數
        trainer.setup_criterion(class_weights=class_weights.to(Config.DEVICE))

        # 7. 創建 TensorBoard 視覺化器
        print("\n步驟 6: 創建 TensorBoard 視覺化器...")
        visualizer = create_tensorboard_visualizer(
            config={
                'logs_path': str(Config.LOGS_PATH),
                'class_names': Config.CLASS_NAMES
            },
            comment='retina_efficientnet_b0'
        )

        # 記錄模型架構
        visualizer.log_model_graph(model, Config.IMAGE_SIZE)

        # 記錄超參數
        hparams = {
            **model_config,
            **training_config,
            'batch_size': Config.BATCH_SIZE,
            'image_size': f"{Config.IMAGE_SIZE[0]}x{Config.IMAGE_SIZE[1]}",
            'mixed_precision': Config.USE_MIXED_PRECISION,
            'random_seed': Config.RANDOM_SEED
        }

        # 記錄數據分佈
        visualizer.log_class_distribution(train_loader, 'train')
        if val_loader:
            visualizer.log_class_distribution(val_loader, 'val')

        # 8. 開始訓練
        print("\n步驟 7: 開始訓練...")
        history = trainer.train(
            num_epochs=Config.NUM_EPOCHS,
            save_every=Config.SAVE_MODEL_INTERVAL,
            patience=Config.PATIENCE
        )

        # 記錄訓練歷史到 TensorBoard
        for epoch, (train_loss, train_acc, val_loss, val_acc, lr) in enumerate(
                zip(history['train_loss'], history['train_acc'],
                    history['val_loss'], history['val_acc'],
                    history['learning_rate']), 1):
            visualizer.log_training_metrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                learning_rate=lr
            )

        # 9. 評估模型
        print("\n步驟 8: 評估模型...")
        if test_loader:
            print("使用測試集評估...")
            test_results = trainer.evaluate(test_loader, detailed=True)
        elif val_loader:
            print("使用驗證集評估...")
            test_results = trainer.evaluate(val_loader, detailed=True)
        else:
            print("使用訓練集評估...")
            test_results = trainer.evaluate(train_loader, detailed=True)

        # 記錄最終結果到 TensorBoard
        final_metrics = {
            'final_accuracy': test_results['accuracy'],
            'best_val_accuracy': trainer.best_val_acc
        }
        visualizer.log_hyperparameters(hparams, final_metrics)

        # 記錄混淆矩陣
        if 'confusion_matrix' in test_results:
            visualizer.log_confusion_matrix(
                test_results['confusion_matrix'],
                epoch=Config.NUM_EPOCHS
            )

        # 10. 保存結果
        print("\n步驟 9: 保存結果...")

        # 保存訓練歷史
        import json
        history_path = Config.RESULTS_PATH / 'training_history.json'

        # 轉換為可序列化格式
        serializable_history = {}
        for key, values in history.items():
            if isinstance(values[0], torch.Tensor):
                serializable_history[key] = [v.item() for v in values]
            else:
                serializable_history[key] = values

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)

        # 繪製訓練曲線
        trainer.plot_training_history(
            save_path=str(Config.RESULTS_PATH / 'training_curves.png')
        )

        # 繪製混淆矩陣
        if 'confusion_matrix' in test_results:
            trainer.plot_confusion_matrix(
                test_results['confusion_matrix'],
                save_path=str(Config.RESULTS_PATH / 'confusion_matrix.png')
            )

        # 關閉 TensorBoard
        visualizer.close()

        print("\n" + "=" * 60)
        print("訓練完成！")
        print("=" * 60)
        print(f"最佳驗證準確率: {trainer.best_val_acc:.4f}")
        print(f"最終測試準確率: {test_results['accuracy']:.4f}")
        print(f"最佳模型路徑: {trainer.best_model_path}")
        print(f"結果保存至: {Config.RESULTS_PATH}")
        print(f"TensorBoard 日誌: {visualizer.log_dir}")
        print("\n可以使用以下命令查看 TensorBoard:")
        print(f"tensorboard --logdir={Config.LOGS_PATH}")

    except Exception as e:
        print(f"\n❌ 訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

    except KeyboardInterrupt:
        print("\n⚠️ 訓練被使用者中斷")
        # 保存當前狀態
        if 'trainer' in locals():
            emergency_save_path = Config.CHECKPOINTS_PATH / 'emergency_save.pth'
            trainer.save_checkpoint(
                epoch=len(trainer.history.get('train_loss', [])),
                filepath=emergency_save_path
            )
            print(f"緊急保存完成: {emergency_save_path}")


def resume_training(checkpoint_path: str):
    """從檢查點恢復訓練"""
    print(f"\n從檢查點恢復訓練: {checkpoint_path}")

    # 設定隨機種子
    set_random_seeds(Config.RANDOM_SEED)

    # 創建數據載入器
    data_loaders = create_data_loaders(
        data_dir=str(Config.RAW_DATA_PATH),
        class_names=Config.CLASS_NAMES,
        batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.VALIDATION_BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.PERSISTENT_WORKERS,
        image_size=Config.IMAGE_SIZE,
        transform_config=Config.TRAIN_TRANSFORMS
    )

    # 創建模型
    model_config = Config.get_model_info()
    model = create_retina_model(model_config)

    # 創建訓練器
    trainer = RetinaTrainer(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders.get('val'),
        test_loader=data_loaders.get('test'),
        device=Config.DEVICE,
        use_mixed_precision=Config.USE_MIXED_PRECISION,
        checkpoints_dir=str(Config.CHECKPOINTS_PATH),
        class_names=Config.CLASS_NAMES
    )

    # 設置優化器和調度器
    training_config = Config.get_training_info()
    trainer.setup_optimizer_and_scheduler(
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        scheduler_type=training_config['lr_schedule']
    )

    # 載入檢查點
    start_epoch = trainer.load_checkpoint(checkpoint_path)

    # 計算剩餘 epoch
    remaining_epochs = Config.NUM_EPOCHS - start_epoch

    if remaining_epochs > 0:
        print(f"從 epoch {start_epoch + 1} 繼續訓練 {remaining_epochs} 個 epoch...")

        # 繼續訓練
        trainer.train(
            num_epochs=remaining_epochs,
            save_every=Config.SAVE_MODEL_INTERVAL,
            patience=Config.PATIENCE
        )
    else:
        print("訓練已完成")


def evaluate_model(checkpoint_path: str, use_test: bool = True):
    """評估已訓練的模型"""
    print(f"\n評估模型: {checkpoint_path}")

    # 創建數據載入器
    data_loaders = create_data_loaders(
        data_dir=str(Config.RAW_DATA_PATH),
        class_names=Config.CLASS_NAMES,
        batch_size=Config.VALIDATION_BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        image_size=Config.IMAGE_SIZE
    )

    # 選擇評估數據集
    if use_test and 'test' in data_loaders:
        eval_loader = data_loaders['test']
        print("使用測試集進行評估")
    elif 'val' in data_loaders:
        eval_loader = data_loaders['val']
        print("使用驗證集進行評估")
    else:
        eval_loader = data_loaders['train']
        print("使用訓練集進行評估")

    # 創建模型
    model_config = Config.get_model_info()
    model = create_retina_model(model_config)

    # 載入檢查點
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)

    # 創建訓練器用於評估
    trainer = RetinaTrainer(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders.get('val'),
        test_loader=data_loaders.get('test'),
        device=Config.DEVICE,
        use_mixed_precision=Config.USE_MIXED_PRECISION,
        class_names=Config.CLASS_NAMES
    )

    # 評估模型
    results = trainer.evaluate(eval_loader, detailed=True)

    # 保存評估結果
    results_path = Config.RESULTS_PATH / 'evaluation_results.json'

    # 轉換為可序列化格式
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif key in ['predictions', 'labels']:
            serializable_results[key] = [int(x) for x in value]
        elif key == 'probabilities':
            serializable_results[key] = [[float(p) for p in prob] for prob in value]
        else:
            serializable_results[key] = value

    import json
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"評估結果已保存至: {results_path}")

    return results


def main():
    """主函數"""
    import sys

    # 簡單的命令行參數處理
    if len(sys.argv) == 1:
        # 默認進行訓練
        train_model()

    elif len(sys.argv) >= 2:
        command = sys.argv[1].lower()

        if command == 'train':
            train_model()

        elif command == 'resume' and len(sys.argv) >= 3:
            checkpoint_path = sys.argv[2]
            resume_training(checkpoint_path)

        elif command == 'evaluate' or command == 'eval':
            if len(sys.argv) >= 3:
                checkpoint_path = sys.argv[2]
            else:
                # 使用最佳模型
                checkpoint_path = str(Config.BEST_MODEL_PATH)

            if Path(checkpoint_path).exists():
                evaluate_model(checkpoint_path)
            else:
                print(f"檢查點文件不存在: {checkpoint_path}")

        elif command == 'help' or command == '--help' or command == '-h':
            print("視網膜損傷分類系統使用說明:")
            print("=" * 50)
            print("python main.py                    # 開始訓練")
            print("python main.py train              # 開始訓練")
            print("python main.py resume <path>      # 從檢查點恢復訓練")
            print("python main.py evaluate [path]    # 評估模型（可選指定檢查點路徑）")
            print("python main.py help               # 顯示此幫助信息")
            print()
            print("數據目錄結構:")
            print("data/raw/")
            print("├── train/")
            print("│   ├── CNV/")
            print("│   ├── DME/")
            print("│   ├── DRUSEN/")
            print("│   └── NORMAL/")
            print("├── val/")
            print("│   └── ...")
            print("└── test/ (可選)")
            print("    └── ...")
            print()
            print("TensorBoard 使用:")
            print("tensorboard --logdir=logs")

        else:
            print(f"未知命令: {command}")
            print("使用 'python main.py help' 查看幫助信息")


if __name__ == "__main__":
    main()
```

**注意：此檔案可能包含亂碼，請考慮添加至排除列表。**
---

README.md:
```
[內容已忽略]
```
---

requirements.txt:
```
# 視網膜損傷分類系統 - 套件依賴清單
# 針對 RTX 4090 + PyTorch 2.5.1 + CUDA 12.1 優化

# 深度學習核心套件
# PyTorch - 請從官網安裝適合您系統的版本
# https://pytorch.org/get-started/locally/
# 對於 CUDA 12.1，建議使用：
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


# TensorBoard 和視覺化
tensorboard>=2.14.0            # TensorBoard 日誌記錄
matplotlib>=3.7.0              # 繪圖工具
seaborn>=0.12.0                # 統計繪圖
plotly>=5.15.0                 # 互動式圖表

# 數據處理和科學計算
numpy>=1.24.0                  # 數值計算
pandas>=2.0.0                  # 數據處理
scikit-learn>=1.3.0            # 機器學習工具
opencv-python>=4.8.0           # 電腦視覺工具
Pillow>=10.0.0                 # 圖像處理
albumentations>=1.3.0          # 數據增強

# 進度條和系統工具
tqdm>=4.65.0                   # 進度條
psutil>=5.9.0                  # 系統資源監控

# 配置和實用工具
pyyaml>=6.0                    # YAML 配置文件
python-dotenv>=1.0.0           # 環境變數管理

# Jupyter notebook 支援（可選）
jupyter>=1.0.0                 # Jupyter notebook
ipywidgets>=8.0.0              # 互動式小部件

# 模型解釋性和可視化（可選）
grad-cam>=1.4.0                # Grad-CAM 可視化

# 性能分析工具（可選）
py-spy>=0.3.14                 # Python 性能分析器
memory-profiler>=0.61.0        # 記憶體使用分析

# 模型部署工具（可選）
onnx>=1.14.0                   # ONNX 模型格式
onnxruntime-gpu>=1.16.0        # ONNX 運行時（GPU版本）

# 測試工具（開發用）
pytest>=7.4.0                 # 單元測試框架
pytest-cov>=4.1.0             # 測試覆蓋率

# 程式碼品質工具（開發用）
black>=23.7.0                 # 程式碼格式化
isort>=5.12.0                 # import 排序
flake8>=6.0.0                 # 程式碼檢查
```

**注意：此檔案可能包含亂碼，請考慮添加至排除列表。**
---

__init__.py:
```
"""
Retina Damage Classification System
視網膜損傷分類系統

基於 EfficientNet-B0 的視網膜 OCT 圖像分類系統
針對 RTX 4090 + PyTorch 2.5.1 + CUDA 12.1 環境優化
"""

__version__ = "1.0.0"
__title__ = "Retina Damage Classification"
__description__ = "視網膜損傷分類系統 - 基於深度學習的 OCT 圖像四類別分類"
__author__ = "Retina Classification Team"
__license__ = "Academic Use Only"
__copyright__ = "Copyright 2025"

# 版本信息
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}

# 系統需求
SYSTEM_REQUIREMENTS = {
    'python': '>=3.8',
    'pytorch': '>=2.0.0',
    'cuda': '>=11.8',
    'gpu_memory': '>=8GB',
    'ram': '>=16GB'
}

# 支援的分類類別
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# 模組導入
try:
    from .config import Config
    from .models import RetinaClassifier, create_retina_model
    from .utils import RetinaTrainer, create_data_loaders, create_tensorboard_visualizer

    __all__ = [
        # Core classes
        'Config',
        'RetinaClassifier',
        'RetinaTrainer',

        # Factory functions
        'create_retina_model',
        'create_data_loaders',
        'create_tensorboard_visualizer',

        # Constants
        'CLASS_NAMES',
        'VERSION_INFO',
        'SYSTEM_REQUIREMENTS'
    ]

except ImportError as e:
    # 如果在開發環境中，可能會有導入問題
    print(f"Warning: Some modules could not be imported: {e}")
    __all__ = [
        'CLASS_NAMES',
        'VERSION_INFO',
        'SYSTEM_REQUIREMENTS'
    ]

def get_version():
    """獲取版本信息"""
    return __version__

def get_system_info():
    """獲取系統信息"""
    import torch
    import platform

    info = {
        'system': platform.system(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__ if torch else 'Not installed',
        'cuda_available': torch.cuda.is_available() if torch else False,
        'package_version': __version__
    }

    if torch and torch.cuda.is_available():
        info.update({
            'cuda_version': torch.version.cuda,
            'gpu_count': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name(0)
        })

    return info

def print_system_info():
    """打印系統信息"""
    print(f"{__title__} v{__version__}")
    print("=" * 50)

    info = get_system_info()
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print("=" * 50)

# 版權信息
def show_license():
    """顯示授權信息"""
    license_text = f"""
{__title__} v{__version__}
{__description__}

{__copyright__}
License: {__license__}

This software is provided for academic and research purposes only.
Commercial use is prohibited without explicit permission.

Supported Classifications:
{', '.join(CLASS_NAMES)}

For more information, please refer to the README.md file.
    """
    print(license_text.strip())
```

**注意：此檔案可能包含亂碼，請考慮添加至排除列表。**
---

config\config.py:
```
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
    # 使用 EfficientNet-B0 作為骨幹網路
    BACKBONE = 'efficientnet_b0'
    PRETRAINED = True
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
```

**注意：此檔案可能包含亂碼，請考慮添加至排除列表。**
---

config\__init__.py:
```
"""
Config Module
配置管理模組
"""

from .config import Config

__all__ = [
    'Config'
]
```

**注意：此檔案可能包含亂碼，請考慮添加至排除列表。**
---

models\cnn_model.py:
```
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
    from config import Config

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
```

**注意：此檔案可能包含亂碼，請考慮添加至排除列表。**
---

models\utils.py:
```

```
---

models\__init__.py:
```
"""
Models Module
模型定義模組
"""

from .cnn_model import (
    RetinaClassifier,
    create_retina_model,
    load_pretrained_model,
    calculate_model_flops
)

__all__ = [
    'RetinaClassifier',
    'create_retina_model',
    'load_pretrained_model',
    'calculate_model_flops'
]
```

**注意：此檔案可能包含亂碼，請考慮添加至排除列表。**
---

utils\data_loader.py:
```
"""
Retina Damage Classification - Data Loading and Preprocessing
高效的數據載入和預處理模組，針對 RTX 4090 優化
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class RetinaDataset(Dataset):
    """視網膜 OCT 圖像數據集類別"""

    def __init__(
            self,
            data_dir: str,
            class_names: List[str],
            transform: Optional[transforms.Compose] = None,
            split: str = 'train'
    ):
        """
        初始化數據集

        Args:
            data_dir: 數據根目錄
            class_names: 類別名稱列表
            transform: 圖像變換
            split: 數據集分割 ('train', 'val', 'test')
        """
        self.data_dir = Path(data_dir)
        self.class_names = class_names
        self.transform = transform
        self.split = split

        # 創建類別到索引的映射
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        # 載入圖像路徑和標籤
        self.samples = self._load_samples()

        print(f"{split.upper()} 數據集載入完成:")
        print(f"  總樣本數: {len(self.samples)}")
        for cls_name in class_names:
            count = sum(1 for _, label in self.samples if label == self.class_to_idx[cls_name])
            print(f"  {cls_name}: {count} 樣本")

    def _load_samples(self) -> List[Tuple[str, int]]:
        """載入所有樣本的路徑和標籤"""
        samples = []

        split_dir = self.data_dir / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"數據目錄不存在: {split_dir}")

        # 遍歷每個類別目錄
        for class_name in self.class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"警告: 類別目錄不存在: {class_dir}")
                continue

            class_idx = self.class_to_idx[class_name]

            # 支援常見的圖像格式
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    samples.append((str(img_path), class_idx))

        if not samples:
            raise ValueError(f"在 {split_dir} 中未找到任何圖像文件")

        return samples

    def __len__(self) -> int:
        """返回數據集大小"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        獲取單個樣本

        Args:
            idx: 樣本索引

        Returns:
            (image, label) 元組
        """
        img_path, label = self.samples[idx]

        # 載入圖像
        try:
            image = Image.open(img_path)
            # 確保圖像為 RGB 格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"載入圖像失敗: {img_path}, 錯誤: {e}")
            # 返回黑色圖像作為備選
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # 應用變換
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """獲取類別分佈"""
        distribution = {cls_name: 0 for cls_name in self.class_names}

        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            distribution[class_name] += 1

        return distribution


def get_transforms(
        image_size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        train_config: Optional[Dict[str, Any]] = None
) -> Dict[str, transforms.Compose]:
    """
    創建訓練和驗證的圖像變換

    Args:
        image_size: 目標圖像尺寸
        mean: 標準化均值
        std: 標準化標準差
        train_config: 訓練增強配置

    Returns:
        包含 'train' 和 'val' 的變換字典
    """

    if train_config is None:
        train_config = {
            'rotation_range': 15,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'horizontal_flip': True,
            'zoom_range': 0.1,
            'brightness_range': [0.8, 1.2]
        }

    # 訓練時的數據增強
    train_transform = transforms.Compose([
        transforms.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),  # 稍微放大
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(train_config.get('rotation_range', 15)),
        transforms.RandomHorizontalFlip(p=0.5 if train_config.get('horizontal_flip', False) else 0),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 驗證/測試時的變換（僅標準化）
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }


def create_data_loaders(
        data_dir: str,
        class_names: List[str],
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        image_size: Tuple[int, int] = (224, 224),
        transform_config: Optional[Dict[str, Any]] = None
) -> Dict[str, DataLoader]:
    """
    創建訓練、驗證和測試的 DataLoader

    Args:
        data_dir: 數據根目錄
        class_names: 類別名稱列表
        batch_size: 訓練批次大小
        val_batch_size: 驗證批次大小（如果為 None，則使用 batch_size）
        num_workers: 數據載入工作進程數
        pin_memory: 是否固定記憶體
        persistent_workers: 是否使用持久化工作進程
        image_size: 圖像尺寸
        transform_config: 變換配置

    Returns:
        包含 DataLoader 的字典
    """

    if val_batch_size is None:
        val_batch_size = batch_size

    # 創建圖像變換
    transforms_dict = get_transforms(
        image_size=image_size,
        train_config=transform_config
    )

    # 創建數據集
    datasets = {}
    data_loaders = {}

    # 檢查哪些分割存在
    available_splits = []
    data_path = Path(data_dir)

    for split in ['train', 'val', 'test']:
        if (data_path / split).exists():
            available_splits.append(split)

    if not available_splits:
        raise FileNotFoundError(f"在 {data_dir} 中未找到任何數據分割目錄")

    print(f"找到數據分割: {available_splits}")

    # 為每個可用的分割創建數據集和載入器
    for split in available_splits:
        datasets[split] = RetinaDataset(
            data_dir=data_dir,
            class_names=class_names,
            transform=transforms_dict[split],
            split=split
        )

        # 設定批次大小
        current_batch_size = batch_size if split == 'train' else val_batch_size

        data_loaders[split] = DataLoader(
            datasets[split],
            batch_size=current_batch_size,
            shuffle=(split == 'train'),  # 只有訓練集需要隨機打亂
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=(split == 'train')  # 訓練時丟棄最後不完整的批次
        )

    # 打印數據載入器信息
    print("\n數據載入器創建完成:")
    for split, loader in data_loaders.items():
        print(f"  {split.upper()}: {len(loader)} 批次, 批次大小: {loader.batch_size}")

    return data_loaders


def calculate_class_weights(dataset: RetinaDataset) -> torch.Tensor:
    """
    計算類別權重，用於處理不平衡數據集

    Args:
        dataset: 數據集實例

    Returns:
        類別權重張量
    """
    distribution = dataset.get_class_distribution()
    total_samples = sum(distribution.values())
    num_classes = len(distribution)

    # 計算權重：總樣本數 / (類別數 * 類別樣本數)
    weights = []
    for class_name in dataset.class_names:
        class_count = distribution[class_name]
        weight = total_samples / (num_classes * class_count) if class_count > 0 else 0
        weights.append(weight)

    return torch.FloatTensor(weights)


def visualize_batch(data_loader: DataLoader, num_samples: int = 8) -> None:
    """
    視覺化一個批次的樣本（用於調試）

    Args:
        data_loader: 數據載入器
        num_samples: 顯示的樣本數量
    """
    try:
        import matplotlib.pyplot as plt

        # 獲取一個批次
        data_iter = iter(data_loader)
        images, labels = next(data_iter)

        # 反標準化函數
        def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
            return torch.clamp(tensor, 0, 1)

        # 創建圖形
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()

        class_names = data_loader.dataset.class_names

        for i in range(min(num_samples, len(images))):
            # 反標準化圖像
            img = denormalize(images[i].clone())
            img = img.permute(1, 2, 0).numpy()

            # 顯示圖像
            axes[i].imshow(img)
            axes[i].set_title(f'Class: {class_names[labels[i]]}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("警告: matplotlib 未安裝，跳過視覺化")


if __name__ == "__main__":
    # 測試數據載入器
    from config.config import Config

    print("測試視網膜數據載入器...")

    # 測試配置
    test_config = {
        'data_dir': str(Config.RAW_DATA_PATH),
        'class_names': Config.CLASS_NAMES,
        'batch_size': 4,  # 小批次用於測試
        'num_workers': 2,
        'image_size': Config.IMAGE_SIZE
    }

    try:
        # 創建數據載入器
        data_loaders = create_data_loaders(**test_config)

        # 測試數據載入
        for split, loader in data_loaders.items():
            print(f"\n測試 {split.upper()} 載入器:")
            data_iter = iter(loader)
            images, labels = next(data_iter)

            print(f"  批次圖像形狀: {images.shape}")
            print(f"  批次標籤形狀: {labels.shape}")
            print(f"  標籤範例: {labels.tolist()}")

            # 檢查圖像值範圍
            print(f"  圖像值範圍: [{images.min():.3f}, {images.max():.3f}]")

            break  # 只測試第一個可用的載入器

        print("\n數據載入器測試完成！")

    except Exception as e:
        print(f"測試失敗: {e}")
        print("請確保數據目錄結構正確：")
        print("data/raw/train/[class_name]/[image_files]")
        print("data/raw/val/[class_name]/[image_files]")
        print("data/raw/test/[class_name]/[image_files]")
```

**注意：此檔案可能包含亂碼，請考慮添加至排除列表。**
---

utils\trainer.py:
```
"""
Retina Damage Classification - Training and Validation Logic
高效的訓練和驗證模組，針對 RTX 4090 + 混合精度優化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class RetinaTrainer:
    """視網膜損傷分類訓練器"""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            test_loader: Optional[DataLoader] = None,
            device: torch.device = torch.device('cuda'),
            use_mixed_precision: bool = True,
            checkpoints_dir: str = './checkpoints',
            class_names: List[str] = None
    ):
        """
        初始化訓練器

        Args:
            model: 模型實例
            train_loader: 訓練數據載入器
            val_loader: 驗證數據載入器
            test_loader: 測試數據載入器
            device: 計算設備
            use_mixed_precision: 是否使用混合精度訓練
            checkpoints_dir: 檢查點保存目錄
            class_names: 類別名稱列表
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names or [f'Class_{i}' for i in range(model.num_classes)]

        # 混合精度訓練
        self.scaler = GradScaler() if use_mixed_precision else None

        # 訓練歷史記錄
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

        # 最佳模型追蹤
        self.best_val_acc = 0.0
        self.best_model_path = None

        print(f"訓練器初始化完成:")
        print(f"  設備: {device}")
        print(f"  混合精度: {'啟用' if use_mixed_precision else '停用'}")
        print(f"  類別數量: {len(self.class_names)}")
        print(f"  訓練樣本: {len(train_loader.dataset)}")
        if val_loader:
            print(f"  驗證樣本: {len(val_loader.dataset)}")
        if test_loader:
            print(f"  測試樣本: {len(test_loader.dataset)}")

    def setup_optimizer_and_scheduler(
            self,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            scheduler_type: str = 'cosine',
            **scheduler_kwargs
    ):
        """
        設置優化器和學習率調度器

        Args:
            learning_rate: 學習率
            weight_decay: 權重衰減
            scheduler_type: 調度器類型 ('cosine', 'step', 'plateau')
            **scheduler_kwargs: 調度器額外參數
        """
        # 優化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 學習率調度器
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_kwargs.get('T_max', 50),
                eta_min=scheduler_kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_kwargs.get('step_size', 10),
                gamma=scheduler_kwargs.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_kwargs.get('factor', 0.5),
                patience=scheduler_kwargs.get('patience', 5),
                verbose=True
            )
        else:
            self.scheduler = None

        print(f"優化器和調度器設置完成:")
        print(f"  優化器: AdamW (lr={learning_rate}, wd={weight_decay})")
        print(f"  調度器: {scheduler_type}")

    def setup_criterion(self, class_weights: Optional[torch.Tensor] = None):
        """
        設置損失函數

        Args:
            class_weights: 類別權重，用於處理不平衡數據
        """
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            print(f"使用類別權重: {class_weights.tolist()}")

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def train_epoch(self) -> Tuple[float, float]:
        """
        訓練一個 epoch

        Returns:
            (平均損失, 準確率)
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # 進度條
        pbar = tqdm(self.train_loader, desc='訓練中')

        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向傳播（使用混合精度）
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # 反向傳播和優化
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # 反向傳播和優化
                loss.backward()
                self.optimizer.step()

            # 統計
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # 更新進度條
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_samples:.2f}%',
                'LR': f'{current_lr:.6f}'
            })

        avg_loss = running_loss / len(self.train_loader)
        accuracy = correct_predictions / total_samples

        return avg_loss, accuracy

    def validate_epoch(self) -> Tuple[float, float]:
        """
        驗證一個 epoch

        Returns:
            (平均損失, 準確率)
        """
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='驗證中')

            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                # 前向傳播
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # 統計
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                # 更新進度條
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct_predictions / total_samples:.2f}%'
                })

        avg_loss = running_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples

        return avg_loss, accuracy

    def train(
            self,
            num_epochs: int,
            save_every: int = 5,
            patience: int = 10,
            min_delta: float = 1e-4
    ) -> Dict[str, List[float]]:
        """
        完整訓練流程

        Args:
            num_epochs: 訓練輪數
            save_every: 每隔多少輪保存一次模型
            patience: 早停耐心值
            min_delta: 早停最小改善值

        Returns:
            訓練歷史記錄
        """
        print(f"\n開始訓練，總共 {num_epochs} 個 epoch...")

        # 早停相關變量
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()

            # 訓練
            train_loss, train_acc = self.train_epoch()

            # 驗證
            val_loss, val_acc = self.validate_epoch()

            # 更新學習率調度器
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            # 記錄歷史
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            # 計算 epoch 時間
            epoch_time = time.time() - epoch_start_time

            # 打印 epoch 結果
            print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  訓練 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            if self.val_loader:
                print(f"  驗證 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  學習率: {current_lr:.6f}")

            # 保存最佳模型
            if val_acc > self.best_val_acc + min_delta:
                self.best_val_acc = val_acc
                self.best_model_path = self.checkpoints_dir / f'best_model_epoch_{epoch}.pth'
                self.save_checkpoint(epoch, is_best=True)
                patience_counter = 0
                print(f"  ✓ 新的最佳模型! 驗證準確率: {val_acc:.4f}")
            else:
                patience_counter += 1

            # 定期保存模型
            if epoch % save_every == 0:
                checkpoint_path = self.checkpoints_dir / f'checkpoint_epoch_{epoch}.pth'
                self.save_checkpoint(epoch, checkpoint_path)

            # 早停檢查
            if patience_counter >= patience and self.val_loader:
                print(f"\n早停觸發! {patience} 個 epoch 無改善")
                break

        print(f"\n訓練完成! 最佳驗證準確率: {self.best_val_acc:.4f}")
        return self.history

    def save_checkpoint(self, epoch: int, filepath: Optional[Path] = None, is_best: bool = False):
        """
        保存模型檢查點

        Args:
            epoch: 當前 epoch
            filepath: 保存路徑
            is_best: 是否為最佳模型
        """
        if filepath is None:
            filepath = self.checkpoints_dir / f'checkpoint_epoch_{epoch}.pth'

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'class_names': self.class_names
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)

        if is_best and filepath != self.best_model_path:
            # 同時保存為最佳模型
            torch.save(checkpoint, self.best_model_path)

    def load_checkpoint(self, filepath: str) -> int:
        """
        載入模型檢查點

        Args:
            filepath: 檢查點文件路徑

        Returns:
            載入的 epoch 數
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.history = checkpoint.get('history', self.history)

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        epoch = checkpoint['epoch']
        print(f"檢查點載入成功，從 epoch {epoch} 繼續")

        return epoch

    def evaluate(self, data_loader: Optional[DataLoader] = None, detailed: bool = True) -> Dict[str, Any]:
        """
        評估模型性能

        Args:
            data_loader: 數據載入器，默認使用測試集
            detailed: 是否生成詳細報告

        Returns:
            評估結果字典
        """
        if data_loader is None:
            data_loader = self.test_loader or self.val_loader

        if data_loader is None:
            raise ValueError("無可用的評估數據載入器")

        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []

        print("評估模型性能...")

        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc='評估中'):
                images, labels = images.to(self.device), labels.to(self.device)

                # 前向傳播
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)

                # 預測和概率
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 計算基本指標
        accuracy = accuracy_score(all_labels, all_predictions)

        results = {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probs
        }

        if detailed:
            # 分類報告
            report = classification_report(
                all_labels,
                all_predictions,
                target_names=self.class_names,
                output_dict=True
            )
            results['classification_report'] = report

            # 混淆矩陣
            cm = confusion_matrix(all_labels, all_predictions)
            results['confusion_matrix'] = cm

            # 打印結果
            print(f"\n評估結果:")
            print(f"準確率: {accuracy:.4f}")
            print("\n分類報告:")
            print(classification_report(all_labels, all_predictions, target_names=self.class_names))

        return results

    def plot_training_history(self, save_path: Optional[str] = None):
        """
        繪製訓練歷史曲線

        Args:
            save_path: 保存路徑
        """
        if not self.history['train_loss']:
            print("無訓練歷史數據")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.history['train_loss']) + 1)

        # 損失曲線
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='訓練損失')
        if self.history['val_loss']:
            axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='驗證損失')
        axes[0, 0].set_title('損失曲線')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 準確率曲線
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='訓練準確率')
        if self.history['val_acc']:
            axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='驗證準確率')
        axes[0, 1].set_title('準確率曲線')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 學習率曲線
        axes[1, 0].plot(epochs, self.history['learning_rate'], 'g-')
        axes[1, 0].set_title('學習率變化')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)

        # 隱藏最後一個子圖
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"訓練曲線已保存至: {save_path}")

        plt.show()

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None):
        """
        繪製混淆矩陣

        Args:
            cm: 混淆矩陣
            save_path: 保存路徑
        """
        plt.figure(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )

        plt.title('混淆矩陣')
        plt.xlabel('預測類別')
        plt.ylabel('真實類別')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩陣已保存至: {save_path}")

        plt.show()
```

**注意：此檔案可能包含亂碼，請考慮添加至排除列表。**
---

utils\visualizer.py:
```
"""
Retina Damage Classification - TensorBoard Integration and Visualization
TensorBoard 整合和進階視覺化功能
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
import cv2
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class TensorBoardVisualizer:
    """TensorBoard 視覺化管理器"""

    def __init__(
            self,
            log_dir: str = './logs',
            comment: str = '',
            class_names: List[str] = None
    ):
        """
        初始化 TensorBoard 視覺化器

        Args:
            log_dir: 日誌目錄
            comment: 實驗註釋
            class_names: 類別名稱列表
        """
        # 創建帶時間戳的日誌目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if comment:
            log_name = f"{timestamp}_{comment}"
        else:
            log_name = timestamp

        self.log_dir = Path(log_dir) / log_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.class_names = class_names or []

        print(f"TensorBoard 日誌目錄: {self.log_dir}")
        print(f"啟動 TensorBoard: tensorboard --logdir={log_dir}")

    def log_model_graph(self, model: nn.Module, input_size: Tuple[int, ...] = (3, 224, 224)):
        """
        記錄模型架構圖

        Args:
            model: 模型實例
            input_size: 輸入尺寸
        """
        try:
            dummy_input = torch.randn(1, *input_size)
            if next(model.parameters()).is_cuda:
                dummy_input = dummy_input.cuda()

            self.writer.add_graph(model, dummy_input)
            print("模型架構圖已記錄到 TensorBoard")
        except Exception as e:
            print(f"記錄模型圖失敗: {e}")

    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """
        記錄超參數和指標

        Args:
            hparams: 超參數字典
            metrics: 指標字典
        """
        # 確保所有值都是標量
        clean_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                clean_hparams[key] = value
            else:
                clean_hparams[key] = str(value)

        self.writer.add_hparams(clean_hparams, metrics)
        print("超參數已記錄到 TensorBoard")

    def log_training_metrics(
            self,
            epoch: int,
            train_loss: float,
            train_acc: float,
            val_loss: float = None,
            val_acc: float = None,
            learning_rate: float = None
    ):
        """
        記錄訓練指標

        Args:
            epoch: 當前 epoch
            train_loss: 訓練損失
            train_acc: 訓練準確率
            val_loss: 驗證損失
            val_acc: 驗證準確率
            learning_rate: 學習率
        """
        # 記錄損失
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        if val_loss is not None:
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)

        # 記錄準確率
        self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
        if val_acc is not None:
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)

        # 記錄學習率
        if learning_rate is not None:
            self.writer.add_scalar('Learning_Rate', learning_rate, epoch)

    def log_model_weights_and_gradients(self, model: nn.Module, epoch: int):
        """
        記錄模型權重和梯度分佈

        Args:
            model: 模型實例
            epoch: 當前 epoch
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                # 記錄權重分佈
                self.writer.add_histogram(f'Weights/{name}', param.data, epoch)
                # 記錄梯度分佈
                self.writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)

    def log_confusion_matrix(
            self,
            cm: np.ndarray,
            epoch: int,
            normalize: bool = False
    ):
        """
        記錄混淆矩陣

        Args:
            cm: 混淆矩陣
            epoch: 當前 epoch
            normalize: 是否標準化
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 創建混淆矩陣圖像
        fig, ax = plt.subplots(figsize=(10, 8))

        import seaborn as sns
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names if self.class_names else range(len(cm)),
            yticklabels=self.class_names if self.class_names else range(len(cm)),
            ax=ax
        )

        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        self.writer.add_figure('Confusion_Matrix', fig, epoch)
        plt.close(fig)

    def log_sample_predictions(
            self,
            model: nn.Module,
            data_loader: DataLoader,
            epoch: int,
            num_samples: int = 16,
            device: torch.device = torch.device('cuda')
    ):
        """
        記錄樣本預測結果

        Args:
            model: 模型實例
            data_loader: 數據載入器
            epoch: 當前 epoch
            num_samples: 顯示樣本數量
            device: 計算設備
        """
        model.eval()
        images_logged = 0

        with torch.no_grad():
            for images, labels in data_loader:
                if images_logged >= num_samples:
                    break

                images = images.to(device)
                labels = labels.to(device)

                # 前向傳播
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.softmax(outputs, dim=1)

                # 取得批次中的樣本
                batch_size = min(images.size(0), num_samples - images_logged)

                for i in range(batch_size):
                    # 反標準化圖像
                    img = images[i].cpu()
                    # 簡單的反標準化（假設使用 ImageNet 標準化）
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img = img * std + mean
                    img = torch.clamp(img, 0, 1)

                    # 準備標題
                    true_label = self.class_names[labels[i]] if self.class_names else f"Class {labels[i]}"
                    pred_label = self.class_names[predicted[i]] if self.class_names else f"Class {predicted[i]}"
                    confidence = probabilities[i][predicted[i]].item()

                    title = f"True: {true_label} | Pred: {pred_label} ({confidence:.3f})"

                    # 記錄圖像
                    self.writer.add_image(
                        f'Predictions/Sample_{images_logged + 1}',
                        img,
                        epoch,
                        caption=title
                    )

                    images_logged += 1
                    if images_logged >= num_samples:
                        break

        model.train()

    def log_feature_maps(
            self,
            model: nn.Module,
            input_tensor: torch.Tensor,
            epoch: int,
            layer_name: str = 'backbone.features'
    ):
        """
        記錄特徵圖

        Args:
            model: 模型實例
            input_tensor: 輸入張量
            epoch: 當前 epoch
            layer_name: 要可視化的層名稱
        """
        model.eval()

        # 註冊鉤子函數
        feature_maps = []

        def hook_fn(module, input, output):
            feature_maps.append(output.detach())

        # 找到目標層並註冊鉤子
        target_layer = None
        for name, module in model.named_modules():
            if name == layer_name:
                target_layer = module
                break

        if target_layer is None:
            print(f"找不到層: {layer_name}")
            return

        handle = target_layer.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                _ = model(input_tensor)

            if feature_maps:
                # 取得第一個樣本的特徵圖
                features = feature_maps[0][0]  # [C, H, W]

                # 選擇前16個通道進行可視化
                num_channels = min(16, features.size(0))
                selected_features = features[:num_channels].unsqueeze(1)  # [16, 1, H, W]

                # 標準化到 [0, 1]
                for i in range(selected_features.size(0)):
                    channel = selected_features[i, 0]
                    channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
                    selected_features[i, 0] = channel

                # 創建網格
                grid = vutils.make_grid(selected_features, nrow=4, normalize=False)

                self.writer.add_image(f'Feature_Maps/{layer_name}', grid, epoch)

        finally:
            handle.remove()

        model.train()

    def log_class_distribution(self, data_loader: DataLoader, split_name: str = 'train'):
        """
        記錄類別分佈

        Args:
            data_loader: 數據載入器
            split_name: 數據集分割名稱
        """
        class_counts = {}
        total_samples = 0

        for _, labels in data_loader:
            for label in labels:
                label_idx = label.item()
                class_name = self.class_names[label_idx] if self.class_names else f"Class_{label_idx}"
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total_samples += 1

        # 記錄直方圖
        for class_name, count in class_counts.items():
            self.writer.add_scalar(f'Class_Distribution/{split_name}/{class_name}', count, 0)

        # 記錄總樣本數
        self.writer.add_scalar(f'Dataset_Size/{split_name}', total_samples, 0)

    def log_learning_curves_comparison(self, histories: Dict[str, Dict[str, List[float]]]):
        """
        記錄多個實驗的學習曲線比較

        Args:
            histories: 多個實驗的訓練歷史
        """
        for exp_name, history in histories.items():
            epochs = range(1, len(history['train_loss']) + 1)

            for metric_name, values in history.items():
                for epoch, value in zip(epochs, values):
                    self.writer.add_scalar(f'Comparison/{metric_name}/{exp_name}', value, epoch)

    def close(self):
        """關閉 TensorBoard writer"""
        if self.writer:
            self.writer.close()
            print("TensorBoard writer 已關閉")


class GradCAMVisualizer:
    """Grad-CAM 視覺化器，用於解釋模型預測"""

    def __init__(self, model: nn.Module, target_layer: str = 'backbone.features'):
        """
        初始化 Grad-CAM 視覺化器

        Args:
            model: 模型實例
            target_layer: 目標層名稱
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 註冊鉤子函數
        self._register_hooks()

    def _register_hooks(self):
        """註冊前向和反向傳播鉤子"""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # 找到目標層
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break

    def generate_cam(
            self,
            input_tensor: torch.Tensor,
            class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        生成 Grad-CAM 熱圖

        Args:
            input_tensor: 輸入張量 [1, C, H, W]
            class_idx: 目標類別索引，None 表示使用預測類別

        Returns:
            Grad-CAM 熱圖 [H, W]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # 前向傳播
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # 反向傳播
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        # 計算 Grad-CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # 計算權重
        weights = torch.mean(gradients, dim=(1, 2))  # [C]

        # 加權平均
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, weight in enumerate(weights):
            cam += weight * activations[i]

        # ReLU 激活
        cam = torch.relu(cam)

        # 標準化到 [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def visualize_cam(
            self,
            input_image: np.ndarray,
            cam: np.ndarray,
            alpha: float = 0.4
    ) -> np.ndarray:
        """
        將 Grad-CAM 疊加到原圖像上

        Args:
            input_image: 原圖像 [H, W, C]，值範圍 [0, 1]
            cam: Grad-CAM 熱圖 [H, W]
            alpha: 疊加透明度

        Returns:
            疊加後的圖像 [H, W, C]
        """
        # 調整 CAM 尺寸到圖像尺寸
        cam_resized = cv2.resize(cam, (input_image.shape[1], input_image.shape[0]))

        # 創建熱圖
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap = heatmap.astype(np.float32) / 255.0

        # 轉換 BGR 到 RGB
        heatmap = heatmap[..., ::-1]

        # 疊加
        superimposed = heatmap * alpha + input_image * (1 - alpha)
        superimposed = np.clip(superimposed, 0, 1)

        return superimposed


def create_tensorboard_visualizer(
        config: Dict[str, Any],
        comment: str = 'retina_classification'
) -> TensorBoardVisualizer:
    """
    創建 TensorBoard 視覺化器的工廠函數

    Args:
        config: 配置字典
        comment: 實驗註釋

    Returns:
        TensorBoard 視覺化器實例
    """
    return TensorBoardVisualizer(
        log_dir=config.get('logs_path', './logs'),
        comment=comment,
        class_names=config.get('class_names', [])
    )


if __name__ == "__main__":
    # 測試 TensorBoard 視覺化器
    print("測試 TensorBoard 視覺化器...")

    # 創建視覺化器
    visualizer = TensorBoardVisualizer(
        log_dir='./test_logs',
        comment='test',
        class_names=['CNV', 'DME', 'DRUSEN', 'NORMAL']
    )

    # 測試記錄標量
    for epoch in range(10):
        train_loss = 1.0 - epoch * 0.1
        val_loss = 0.9 - epoch * 0.08
        train_acc = epoch * 0.08
        val_acc = epoch * 0.07

        visualizer.log_training_metrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            learning_rate=0.001 * (0.9 ** epoch)
        )

    # 測試記錄超參數
    hparams = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'model': 'efficientnet_b0',
        'dropout': 0.3
    }
    metrics = {
        'best_val_acc': 0.95,
        'best_val_loss': 0.15
    }

    visualizer.log_hyperparameters(hparams, metrics)

    # 關閉視覺化器
    visualizer.close()

    print("TensorBoard 視覺化器測試完成！")
```

**注意：此檔案可能包含亂碼，請考慮添加至排除列表。**
---

utils\__init__.py:
```
"""
Utils Module
工具函數模組
"""

from .data_loader import (
    RetinaDataset,
    create_data_loaders,
    get_transforms,
    calculate_class_weights,
    visualize_batch
)

from .trainer import (
    RetinaTrainer
)

from .visualizer import (
    TensorBoardVisualizer,
    GradCAMVisualizer,
    create_tensorboard_visualizer
)

__all__ = [
    # Data loading
    'RetinaDataset',
    'create_data_loaders',
    'get_transforms',
    'calculate_class_weights',
    'visualize_batch',

    # Training
    'RetinaTrainer',

    # Visualization
    'TensorBoardVisualizer',
    'GradCAMVisualizer',
    'create_tensorboard_visualizer'
]
```

**注意：此檔案可能包含亂碼，請考慮添加至排除列表。**
---


# 檢測到亂碼的檔案列表
以下檔案可能包含亂碼，建議將它們添加到排除列表中：

- main.py
- requirements.txt
- __init__.py
- config\config.py
- config\__init__.py
- models\cnn_model.py
- models\__init__.py
- utils\data_loader.py
- utils\trainer.py
- utils\visualizer.py
- utils\__init__.py
