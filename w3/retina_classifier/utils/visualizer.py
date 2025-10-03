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