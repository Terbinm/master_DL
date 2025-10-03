# inference_app.py
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import gradio as gr
from pathlib import Path

# 1) 匯入你訓練時的模型結構
from teacher_ipynb.SimpleCNN import SimpleCNN

# 2) 基本設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = Path("./best_gender_model_v1.pth")  # 與你訓練程式一致
class_names = ["DRUSEN", "NORMAL"]              # 和訓練時的輸出順序一致

# 3) 影像前處理（與 eval_transform 一致）
eval_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# 4) 載入模型做推論
@torch.no_grad()
def predict(img: Image.Image):
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(device)   # [1,3,224,224]

    # 懶載入（第一次呼叫時才載入權重，之後重用）
    if not hasattr(predict, "model"):
        model = SimpleCNN().to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        predict.model = model

    logits = predict.model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # [2]
    top_idx = int(probs.argmax())
    label = class_names[top_idx]

    # 回傳給 Gradio：一個分類標籤（含機率）+ 原圖回顯
    # gr.Label 若收到 {class: prob} 會顯示排序條
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}, img

# 5) Gradio 介面
with gr.Blocks(title="Retina Classifier") as demo:
    gr.Markdown("## 視網膜影像分類（DRUSEN vs NORMAL）\n上傳影像後自動推論。")
    with gr.Row():
        inp = gr.Image(type="pil", label="上傳影像（RGB）")
        with gr.Column():
            out_pred = gr.Label(num_top_classes=2, label="分類結果（機率）")
            out_view = gr.Image(label="原圖回顯")
    btn = gr.Button("開始判斷")
    btn.click(predict, inputs=inp, outputs=[out_pred, out_view])
    inp.change(predict, inputs=inp, outputs=[out_pred, out_view])  # 拖拉自動跑

# 若要臨時對外給同事看，可改 demo.launch(share=True)
demo.launch()
