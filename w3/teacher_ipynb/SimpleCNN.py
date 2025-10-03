import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),  # 卷積層，3通道→32通道
            nn.BatchNorm2d(32),                                    # 批次正規化
            nn.ReLU(),                                             # 激活
            nn.MaxPool2d(2, 2),                                    # 最大池化

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),                              # Dropout
            nn.Linear(256 * 14 * 14, 256),                         # 全連接層，展平後→256維
            nn.ReLU(),                                             # 激活
            nn.Dropout(dropout_rate),                              # Dropout
            nn.Linear(256, 2)                                      # 全連接層，256→2類
        )

    def forward(self, x):
        x = self.features(x)
        # x.view(x.size(0), -1) 的用法與意義：
        # x.size(0)：取得 batch_size，保持每個 batch 的樣本數不變
        # -1：自動計算剩餘維度，將所有 channel 與空間維度展平成一維
        # 例如 x shape (batch_size, 256, 14, 14) 變成 (batch_size, 256*14*14)
        x = x.view(x.size(0), -1)  # 展平多維特徵圖為二維。或是用 x = x.flatten(1)  從第1維開始展平，效果相同
        x = self.classifier(x)
        return x

    '''
    x = x.view(x.size(0), -1) 為什麼要這樣做？
    這是為了將多維的特徵圖展平成二維，以便輸入到全連接層：
    卷積層輸出：4維 tensor (batch_size, channels, height, width)
    全連接層需要：2維 tensor (batch_size, features)
    所以 view() 的作用是：
    保持 batch_size 維度
    將後面的所有維度（channels × height × width）壓縮成一個維度
    '''