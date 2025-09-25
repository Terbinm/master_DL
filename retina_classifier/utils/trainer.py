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