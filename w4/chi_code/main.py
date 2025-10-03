from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch, random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from time import time
from collections import defaultdict

# 換成 TransferCNN
from transfer_cnn import TransferCNN

# 記錄每個 epoch 的結果
history = defaultdict(list)

# 列印 & 存 history
def log_epoch(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc, dt):
    print(f"Epoch {epoch:02d}/{num_epochs} | "
          f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
          f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} | dt:{dt:.1f}s")
    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["dt"].append(dt)


# ---------------------------------------
# 資料集設定
# ---------------------------------------
root = "C:\\led_code\\master_DL\\data\\retina_classifier\\raw"
train_dir = f"{root}\\train"
val_dir   = f"{root}\\val"
test_dir  = f"{root}\\test"

data_limit = 1000  # 每類別最多取多少張圖

# 資料增強
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

if __name__ == '__main__':
    # 讀取資料
    train_full = datasets.ImageFolder(train_dir, transform=train_transform)
    val_full   = datasets.ImageFolder(val_dir,   transform=eval_transform)
    test_full  = datasets.ImageFolder(test_dir,  transform=eval_transform)

    print("classes:", train_full.classes)
    print("class_to_idx:", train_full.class_to_idx)
    print("sizes:", len(train_full), len(val_full), len(test_full))

    # 選 GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 只保留 DRUSEN 和 NORMAL
    keep_classes = ['DRUSEN', 'NORMAL']
    cls2idx = train_full.class_to_idx
    keep_idx = set(cls2idx[c] for c in keep_classes if c in cls2idx)

    def filter_subset(imgfolder, limit=None):
        kept = [i for i, (_, y) in enumerate(imgfolder.samples) if y in keep_idx]
        if limit is not None and len(kept) > limit:
            kept = random.sample(kept, limit)
        return Subset(imgfolder, kept)

    train_ds = filter_subset(train_full, limit=data_limit)
    val_ds   = filter_subset(val_full, limit=data_limit)
    test_ds  = filter_subset(test_full, limit=data_limit)

    batch_size = 64
    num_workers = 2
    prefetch_factor = 2
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              prefetch_factor=prefetch_factor)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              prefetch_factor=prefetch_factor)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              prefetch_factor=prefetch_factor)

    print("篩選後 sizes:", len(train_ds), len(val_ds), len(test_ds))


    # ---------------------------------------
    # 建立模型 (TransferCNN)
    # ---------------------------------------
    model = TransferCNN(num_classes=2, freeze_backbone=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    num_epochs = 50
    best_val_acc = 0.0
    save_path = Path("./best_retina_model.pth")

    # ---------------------------------------
    # 訓練流程
    # ---------------------------------------
    def run_one_epoch(model, loader, train=True):
        if train:
            model.train()
        else:
            model.eval()
        total, correct, running_loss = 0, 0, 0.0
        all_logits, all_labels = [], []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                logits = model(xb)
                loss = criterion(logits, yb)
                if train:
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(yb.detach().cpu())
        avg_loss = running_loss / total
        acc = correct / total
        return avg_loss, acc

    print("開始訓練...")
    early_stopping_time = 10
    early_stopping_count = early_stopping_time

    for epoch in range(1, num_epochs + 1):
        t0 = time()
        train_loss, train_acc = run_one_epoch(model, train_loader, train=True)
        val_loss, val_acc = run_one_epoch(model, val_loader, train=False)
        dt = time() - t0
        log_epoch(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc, dt)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved new best to {save_path} (val_acc={best_val_acc:.4f})")
            early_stopping_count = early_stopping_time
        else:
            early_stopping_count -= 1
            if early_stopping_count == 0:
                print("Early stopping...")
                break

    print("best_val_acc:", best_val_acc)

    # ---------------------------------------
    # 測試集驗證
    # ---------------------------------------
    state = torch.load(save_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_logits, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu()
            all_logits.append(logits.cpu())
            all_preds.append(preds)
            all_labels.append(yb)

    all_logits = torch.cat(all_logits)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = (all_preds == all_labels).float().mean().item()
    print(f"[TEST] accuracy: {acc:.4f}")
    print("[TEST] classification_report:")
    print(classification_report(all_labels.numpy(), all_preds.numpy(), target_names=keep_classes))

    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())
    print("[TEST] confusion_matrix:\n", cm)

    # ---------------------------------------
    # 繪圖
    # ---------------------------------------
    epochs = history["epoch"]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], keep_classes)
    plt.yticks([0, 1], keep_classes)
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, str(z), ha='center', va='center')
    plt.show()

    # ---------------------------------------
    # ROC & AUC
    # ---------------------------------------
    probs = F.softmax(all_logits, dim=1)[:, 1].numpy()  # 取 NORMAL 類的機率
    auc = roc_auc_score(all_labels.numpy(), probs)
    print(f"[TEST] ROC-AUC (NORMAL as positive): {auc:.4f}")

    fpr, tpr, _ = roc_curve(all_labels.numpy(), probs)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.title("ROC Curve (NORMAL positive)")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()
