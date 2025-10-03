import time, torch

def benchmark_loader(loader, n_warmup=10, n_batches=50):
    # 丟幾個 warmup 批次避免冷啟動影響
    it = iter(loader)
    for _ in range(n_warmup):
        xb, yb = next(it)
        # 模擬搬到 GPU（如果你實際訓練時會搬）
        xb = xb.cuda(non_blocking=True) if torch.cuda.is_available() else xb

    # 正式計時
    t0 = time.time()
    cnt = 0
    for _ in range(n_batches):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb = next(it)
        xb = xb.cuda(non_blocking=True) if torch.cuda.is_available() else xb
        cnt += xb.size(0)
    t1 = time.time()
    print(f"samples/sec ≈ {cnt / (t1 - t0):.1f}")

# 逐一嘗試 num_workers = 0,2,4,6,8 並記錄 samples/sec，選最高者
