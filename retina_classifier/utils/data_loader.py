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