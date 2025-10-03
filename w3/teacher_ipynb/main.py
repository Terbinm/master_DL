from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch,random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from time import time


from teacher_ipynb.SimpleCNN import SimpleCNN
from collections import defaultdict

history = defaultdict(list)  # epoch-wise logs

def log_epoch(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc, dt):
    """同步：列印 & 存成 history 供之後繪圖"""
    print(f"Epoch {epoch:02d}/{num_epochs} | "
          f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
          f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} | dt:{dt:.1f}s")
    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["dt"].append(dt)

# dir ------------------------------------------------------------------------------------------------------------------
root = "C:\\led_code\\master_DL\\retina_classifier\\data\\raw2"
train_dir = f"{root}\\train"
val_dir   = f"{root}\\val"
test_dir  = f"{root}\\test"

data_limit = 1000  # 每類別最多取多少張圖，None 表示不限制
# transforms(資料增強+normalization) # ---------------------------------------------------------
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
    #  Read dataset---------------------------------------------------------
    train_full = datasets.ImageFolder(train_dir, transform=train_transform)
    val_full   = datasets.ImageFolder(val_dir,   transform=eval_transform)
    test_full  = datasets.ImageFolder(test_dir,  transform=eval_transform)

    print("classes:", train_full.classes)
    print("class_to_idx:", train_full.class_to_idx)
    print("sizes:", len(train_full), len(val_full), len(test_full))

    # ---- 4090 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- 篩選要的，其他丟掉 ----
    keep_classes = ['DRUSEN', 'NORMAL']
    cls2idx = train_full.class_to_idx
    keep_idx = set(cls2idx[c] for c in keep_classes if c in cls2idx)


    def filter_subset(imgfolder, limit=None):
        kept = [i for i, (_, y) in enumerate(imgfolder.samples) if y in keep_idx]
        if limit is not None and len(kept) > limit:
            kept = random.sample(kept, limit)  # 隨機抽 limit 筆
        return Subset(imgfolder, kept)


    train_ds = filter_subset(train_full, limit=data_limit)
    val_ds = filter_subset(val_full, limit=data_limit)
    test_ds = filter_subset(test_full, limit=data_limit)


    # ---- DataLoader（維持你原本的 batch_size=48，可調整）----
    batch_size = 64
    num_workers = 4  # Windows/本機可先用 0~2，如果有問題就設 0
    prefetch_factor = 4 # 提前N個epoch準備
    persistent_workers = True  # workers 過勞死，不休息
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

    print("篩選後 sizes:", len(train_ds), len(val_ds), len(test_ds))

    # ---- 快速檢查：類別分布 & 一個 batch 形狀 ----
    def _count_targets_sum(type,targets):
        return sum(t == cls2idx[type] for t in targets)


    def count_targets(imgfolder_subset):
        # 取出原 ImageFolder 的 targets，依 subset 的 indices 統計
        base = imgfolder_subset.dataset  # ImageFolder
        idxs = imgfolder_subset.indices
        targets = [base.targets[i] for i in idxs]
        return (_count_targets_sum('DRUSEN',targets),
                _count_targets_sum('NORMAL',targets))

    train_drusen, train_normal = count_targets(train_ds)
    # val_drusen, val_normal     = count_targets(val_ds)
    test_drusen, test_normal   = count_targets(test_ds)

    print(f"train  -> DRUSEN: {train_drusen}, NORMAL: {train_normal}, total: {len(train_ds)}")
    # print(f"val    -> DRUSEN: {val_drusen},   NORMAL: {val_normal},   total: {len(val_ds)}")
    print(f"test   -> DRUSEN: {test_drusen},  NORMAL: {test_normal},  total: {len(test_ds)}")

    # 拿一個 batch 看尺寸與標籤 (超花時間)
    # xb, yb = next(iter(train_loader))
    # print("batch shape:", xb.shape, "labels example:", yb[:16].tolist())

    # model build -----------------------------------------------------------------------------------------------------
    model = SimpleCNN().to(device) # model
    criterion = nn.CrossEntropyLoss() # Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) # Optimizer

    num_epochs = 100 # 訓練幾個 epoch ------------------------------------------------------------------------------------
    best_val_acc = 0.0 #(有可能是val_acc或是test_acc)
    save_path = Path("./best_gender_model_v1.pth")  #


    def run_one_epoch(model, loader, train=True):
        if train:
            model.train()
        else:
            model.eval()
        total, correct, running_loss = 0, 0, 0.0
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
        avg_loss = running_loss / total
        acc = correct / total
        return avg_loss, acc


    print("開始epoch訓練...")
    early_stopping_time = 10
    early_stopping_count = early_stopping_time
    # --- 訓練主迴圈 ---
    for epoch in range(1, num_epochs + 1):
        t0 = time()
        train_loss, train_acc = run_one_epoch(model, train_loader, train=True)
        # val_loss, val_acc = run_one_epoch(model, val_loader, train=False)
        test_loss, test_acc = run_one_epoch(model, test_loader, train=False)
        dt = time() - t0
        log_epoch(epoch, num_epochs, train_loss, train_acc, test_loss, test_acc, dt)

        # 儲存最佳模型（以 val_acc 為準）
        # if val_acc > best_val_acc:
        if test_acc > best_val_acc:
            best_val_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved new best to {save_path} (val_acc={best_val_acc:.4f})")
            early_stopping_count = early_stopping_time
        else:
            early_stopping_count -= 1
            if early_stopping_count == 0:
                print("Early stopping...")
                break

    print("best_val_acc:", best_val_acc)
    # 訓練完成!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # 載入最佳權重 ------------------------------------------------------------------------------------------------------
    best_path = Path("./best_gender_model_v1.pth")
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 蒐集 test 預測 ---------------------------------------------------------------------------------------------------
    all_logits = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu()
            all_logits.append(logits.cpu())
            all_preds.append(preds)
            all_labels.append(yb)

    all_logits = torch.cat(all_logits, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 指標：accuracy / classification report / confusion matrix --------------------------------------------------------
    acc = (all_preds == all_labels).float().mean().item()
    print(f"[TEST] accuracy: {acc:.4f}")

    target_names = ["DRUSEN", "NORMAL"]  # 對應你前面顯示的 classes 順序
    print("\n[TEST] classification_report:")
    print(classification_report(all_labels.numpy(), all_preds.numpy(), target_names=target_names, digits=4))

    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())
    print("[TEST] confusion_matrix (rows=true, cols=pred):\n", cm)

    # ====== 訓練完成後：繪圖 ======
    # 1) Loss / Acc 曲線（兩張圖）
    epochs = history["epoch"]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("acc_curve.png", dpi=150)
    plt.show()

    # 2) 每個 epoch 花費時間（選用）
    plt.figure(figsize=(6, 3.5))
    plt.plot(epochs, history["dt"], marker="o")
    plt.title("Seconds per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.tight_layout()
    plt.savefig("time_per_epoch.png", dpi=150)
    plt.show()

    # 4) 畫混淆矩陣（matplotlib，無 seaborn）
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], target_names)
    plt.yticks([0, 1], target_names)
    # 在格子內標數字
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, str(z), ha='center', va='center')
    plt.tight_layout()
    plt.show()

    # 5) 進階：ROC-AUC（二分類可用）
    probs = F.softmax(all_logits, dim=1)[:, 1].numpy()  # 取 NORMAL 類的機率（index=1）
    auc = roc_auc_score(all_labels.numpy(), probs)
    print(f"[TEST] ROC-AUC (NORMAL as positive): {auc:.4f}")

    fpr, tpr, _ = roc_curve(all_labels.numpy(), probs)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--')
    plt.title("ROC Curve (NORMAL positive)")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()
    plt.show()





