# 視網膜損傷分類系統

基於 **EfficientNet-B0** 的視網膜 OCT 圖像分類系統，針對 **RTX 4090 + PyTorch 2.5.1 + CUDA 12.1** 環境優化，實現四類視網膜疾病的高精度分類。

## 🎯 系統特點

- **高性能**: 基於 EfficientNet-B0 輕量級架構
- **混合精度訓練**: 充分利用 RTX 4090 的 Tensor Cores，加速訓練並節省顯存
- **完整監控**: 集成 TensorBoard 實時監控訓練過程
- **模組化設計**: 清晰的代碼結構，易於維護和擴展
- **最小系統**: 避免過度複雜化，專注核心功能實現

## 📋 分類類別

- **CNV**: 脈絡膜新血管化 (Choroidal Neovascularization)
- **DME**: 糖尿病性黃斑水腫 (Diabetic Macular Edema)
- **DRUSEN**: 玻璃膜疣 (Drusen)
- **NORMAL**: 正常視網膜

## 🏗️ 系統架構

```
retina_classifier/
├── config/
│   └── config.py                 # 統一配置管理
├── data/
│   ├── raw/                      # 原始數據
│   └── processed/                # 預處理數據
├── models/
│   ├── cnn_model.py             # EfficientNet-B0 模型
│   └── utils.py                 # 模型工具
├── utils/
│   ├── data_loader.py           # 數據載入和增強
│   ├── trainer.py               # 訓練和驗證邏輯
│   └── visualizer.py            # TensorBoard 整合
├── checkpoints/                  # 模型檢查點
├── logs/                        # TensorBoard 日誌
├── results/                     # 結果輸出
├── main.py                      # 主程式入口
├── requirements.txt             # 依賴套件
└── README.md                    # 使用說明
```

## 🚀 快速開始

### 1. 環境需求

- **Python**: 3.8+
- **GPU**: NVIDIA RTX 4090 或同等級 GPU
- **CUDA**: 12.1
- **RAM**: 16GB+
- **VRAM**: 8GB+ (建議 16GB+)

### 2. 安裝套件

```bash
# 1. 從 PyTorch 官網安裝適合的版本
# https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. 安裝其他依賴套件
pip install -r requirements.txt
```

### 3. 數據準備

將您的 OCT 圖像按以下結構組織：

```
data/raw/
├── train/
│   ├── CNV/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── DME/
│   │   └── ...
│   ├── DRUSEN/
│   │   └── ...
│   └── NORMAL/
│       └── ...
├── val/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
└── test/ (可選)
    ├── CNV/
    ├── DME/
    ├── DRUSEN/
    └── NORMAL/
```

**注意事項**:
- 支持的圖像格式：`.jpg`, `.jpeg`, `.png`, `.bmp`
- 圖像會自動調整為 224×224 像素
- 建議每類至少 100 張圖像用於訓練

### 4. 開始訓練

```bash
# 基本訓練
python main.py

# 或明確指定訓練命令
python main.py train
```

### 5. 監控訓練過程

訓練開始後，在新的終端視窗啟動 TensorBoard：

```bash
tensorboard --logdir=logs
```

然後在瀏覽器中打開 `http://localhost:6006` 查看訓練曲線、模型架構等。

## 📊 系統配置

### 核心配置 (config/config.py)

```python
# 硬體配置 (針對 RTX 4090 優化)
BATCH_SIZE = 64                    # 可調整至 128
NUM_WORKERS = 8                    # CPU 核心數
USE_MIXED_PRECISION = True         # 啟用混合精度

# 模型配置
BACKBONE = 'efficientnet_b0'       # 骨幹網路
NUM_CLASSES = 4                    # 分類數量
DROPOUT_RATE = 0.3                 # Dropout 比例

# 訓練配置
NUM_EPOCHS = 50                    # 訓練輪數
LEARNING_RATE = 1e-3               # 學習率
PATIENCE = 10                      # 早停耐心值
```

### 自定義配置

如需修改配置，請編輯 `config/config.py` 文件中的相應參數。

## 💻 使用方法

### 基本命令

```bash
# 開始訓練
python main.py train

# 從檢查點恢復訓練
python main.py resume checkpoints/checkpoint_epoch_20.pth

# 評估模型（使用最佳模型）
python main.py evaluate

# 評估指定檢查點
python main.py evaluate checkpoints/best_model.pth

# 顯示幫助信息
python main.py help
```

### 進階功能

#### 1. 自定義數據增強

編輯 `config/config.py` 中的 `TRAIN_TRANSFORMS`:

```python
TRAIN_TRANSFORMS = {
    'rotation_range': 20,           # 旋轉角度
    'horizontal_flip': True,        # 水平翻轉
    'brightness_range': [0.8, 1.2], # 亮度變化
    'zoom_range': 0.15             # 縮放範圍
}
```

#### 2. 模型微調

```python
# 在 trainer.py 中解凍部分層進行微調
trainer.model.unfreeze_backbone(num_layers=3)
```

#### 3. 學習率調度

支持多種學習率調度策略：

- `cosine`: 餘弦退火
- `step`: 階段式衰減  
- `plateau`: 自適應衰減

## 📈 性能指標

### 預期性能 (RTX 4090)

- **訓練準確率**: 95%+
- **驗證準確率**: 90%+
- **訓練時間**: 1-2 小時 (50 epochs)
- **推理速度**: ~100 images/sec
- **模型大小**: ~25 MB

### 性能優化建議

1. **批次大小**: RTX 4090 建議使用 64-128
2. **數據載入**: 使用 8-12 個工作進程
3. **混合精度**: 必須啟用以充分利用 Tensor Cores
4. **內存固定**: 啟用 `pin_memory=True`

## 🔧 故障排除

### 常見問題

#### 1. CUDA 記憶體不足

```bash
# 解決方案：
# 1. 減少批次大小
BATCH_SIZE = 32

# 2. 啟用梯度累積
# 在 trainer.py 中修改相關代碼
```

#### 2. 數據載入錯誤

```bash
# 檢查數據結構
python -c "from utils.data_loader import *; print('數據結構檢查通過')"

# 檢查圖像格式
find data/raw -name "*.jpg" | head -5
```

#### 3. TensorBoard 無法啟動

```bash
# 確認安裝
pip install tensorboard

# 指定端口
tensorboard --logdir=logs --port=6007
```

### 性能調優

#### 1. 記憶體使用優化

```python
# 在 config.py 中調整
NUM_WORKERS = 4                    # 減少數據載入進程
BATCH_SIZE = 32                    # 減少批次大小
```

#### 2. 訓練速度優化

```python
# 啟用所有性能優化
USE_MIXED_PRECISION = True         # 混合精度
PIN_MEMORY = True                  # 內存固定
PERSISTENT_WORKERS = True          # 持久化工作進程
```

## 📁 輸出文件

### 訓練完成後的輸出

- `checkpoints/best_model.pth`: 最佳模型權重
- `results/training_curves.png`: 訓練曲線圖
- `results/confusion_matrix.png`: 混淆矩陣
- `results/training_history.json`: 訓練歷史數據
- `logs/`: TensorBoard 日誌文件

### 模型部署

```python
# 載入最佳模型進行推理
from models.cnn_model import load_pretrained_model
from config.config import Config

model = load_pretrained_model(
    'checkpoints/best_model.pth',
    Config.get_model_info()
)
```

## 🔬 模型解釋性

### Grad-CAM 可視化

```python
from utils.visualizer import GradCAMVisualizer

# 創建 Grad-CAM 視覺化器
grad_cam = GradCAMVisualizer(model)

# 生成熱圖
cam = grad_cam.generate_cam(input_tensor)
visualization = grad_cam.visualize_cam(original_image, cam)
```

## 📚 技術細節

### 模型架構

- **骨幹網路**: EfficientNet-B0 (預訓練於 ImageNet)
- **參數量**: ~5.3M
- **分類頭**: 全局平均池化 + 兩層全連接
- **正則化**: Dropout (0.3) + 權重衰減

### 數據處理

- **圖像尺寸**: 224×224 像素
- **標準化**: ImageNet 統計數據
- **增強技術**: 旋轉、翻轉、亮度調整、縮放

### 訓練策略

- **優化器**: AdamW
- **損失函數**: 交叉熵損失（支持類別權重）
- **學習率調度**: 餘弦退火
- **早停機制**: 監控驗證準確率


**享受您的視網膜損傷分類研究！** 🎉