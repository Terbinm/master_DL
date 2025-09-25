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
        visualizer.log_model_graph(model, input_size=(3, *Config.IMAGE_SIZE))

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

            visualizer.writer.flush()

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