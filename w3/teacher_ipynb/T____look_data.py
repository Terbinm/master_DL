from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import torch.nn as nn
import numpy as np


import matplotlib.pyplot as plt

from teacher_ipynb.SimpleCNN import SimpleCNN

# 圖片都是L(灰階圖)
root = "C:\\led_code\\master_DL\\retina_classifier\\data\\raw"
train_dir = f"{root}/train"
val_dir   = f"{root}/val"
test_dir  = f"{root}/test"


if True:

    # 看圖片
    cnv_img_sample_img = os.path.join(train_dir,"cnv","CNV-81630-4.jpeg")
    image = Image.open(cnv_img_sample_img)

    # 看圖片
    plt.imshow(image)
    plt.axis("off")  # 隱藏座標軸
    plt.show()

    # 看維度
    arr = np.array(image)
    print(arr.shape)
    print(arr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備:{device}")
    model = SimpleCNN().to(device)








