
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random

from utils.config_loader import load_config
from utils.dataLoader import get_loader
from models.perceiver import MultimodalPerceiver

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------- 配置 ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
cfg = load_config(BASE_DIR / "config/config.yaml")
DEVICE = torch.device(cfg.training.device)
SEED = cfg.training.seed
set_seed(SEED)

BATCH_SIZE = cfg.data.batch_size
EPOCHS = cfg.training.epochs
LR = cfg.training.lr

PKL_PATH = Path("D:/MyProject/RobotPerceiver/data/open_dataset/mosi/Processed/aligned_50.pkl")
ALPHA_PATH = Path("./data/alpha/alpha_masks.npz")

# ---------------- 数据加载 ----------------
train_loader = get_loader(PKL_PATH, "train", batch_size=BATCH_SIZE, shuffle=True)
val_loader   = get_loader(PKL_PATH, "valid", batch_size=BATCH_SIZE, shuffle=False)
test_loader  = get_loader(PKL_PATH, "test", batch_size=BATCH_SIZE, shuffle=False)

# ---------------- 模型初始化 ----------------
model = MultimodalPerceiver(cfg).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ---------------- 加载 HSIC alpha（只在训练集使用） ----------------
if ALPHA_PATH.exists():
    alpha_data = np.load(ALPHA_PATH)
    alpha_text   = torch.tensor(alpha_data["text"], dtype=torch.float32, device=DEVICE)
    alpha_audio  = torch.tensor(alpha_data["audio"], dtype=torch.float32, device=DEVICE)
    alpha_vision = torch.tensor(alpha_data["vision"], dtype=torch.float32, device=DEVICE)
    print("Loaded HSIC alpha masks from file.")
else:
    alpha_text = alpha_audio = alpha_vision = None
    print("Alpha masks not found. Run compute_alpha_mask.py first.")

model.alpha_text = alpha_text
model.alpha_audio = alpha_audio
model.alpha_vision = alpha_vision

# ---------------- 日志文件夹 ----------------
log_root = Path(f"D:/MyProject/RobotPerceiver/log/per-hsic-mosi/seed{SEED}")
log_root.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
txt_file = log_root / f"training_log_{timestamp}.txt"
csv_file = log_root / f"training_log_{timestamp}.csv"

# 写入 CSV 表头
with open(csv_file, mode="w", newline="") as f_csv:
    writer = pd.DataFrame(columns=[
        "Epoch", "Train_Loss", "Train_Acc", "Train_Precision", "Train_Recall", "Train_F1",
        "Val_Loss", "Val_Acc", "Val_Precision", "Val_Recall", "Val_F1"
    ])
    writer.to_csv(f_csv, index=False)

def log(s):
    print(s)
    with open(txt_file, "a") as f:
        f.write(s + "\n")

results = []

# ---------------- 训练循环 ----------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    all_preds, all_labels = [], []

    for text, audio, vision, y in train_loader:
        text, audio, vision, y = text.to(DEVICE), audio.to(DEVICE), vision.to(DEVICE), y.to(DEVICE).long()
        optimizer.zero_grad()

        # ---------------- Forward ----------------
        t_proj = model.text_proj(text)
        a_proj = model.audio_proj(audio)
        v_proj = model.vision_proj(vision)

        '''
            这一块得修改一下，然后再跑代码试试
            因为我的目标是：第一次 cross-attention 使用 alpha 指导 latent 关注哪些特征，不改变 x 本身
            可这段代码是：alpha 直接乘到 x 上，相当于修改了 x，本质上是“静态加权”，不是“引导 attention”
            这不是我想要的
        '''
        # 训练阶段使用 alpha
        if model.alpha_text is not None:
            B, T, D_latent = t_proj.shape
            mask_text   = model.alpha_text.unsqueeze(0).unsqueeze(1).expand(B, T, D_latent)
            mask_audio  = model.alpha_audio.unsqueeze(0).unsqueeze(1).expand(B, T, D_latent)
            mask_vision = model.alpha_vision.unsqueeze(0).unsqueeze(1).expand(B, T, D_latent)
            fused = torch.cat([t_proj*mask_text, a_proj*mask_audio, v_proj*mask_vision], dim=-1)
        else:
            fused = torch.cat([t_proj, a_proj, v_proj], dim=-1)

        fused = nn.functional.layer_norm(fused, fused.shape[-1:])
        output = model.perceiver(fused)

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * y.size(0)
        all_preds.append(output.argmax(dim=1).cpu())
        all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    train_acc = accuracy_score(all_labels, all_preds)
    precision, recall, train_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    avg_train_loss = train_loss / len(train_loader.dataset)

    log(f"Epoch {epoch+1}/{EPOCHS}\n"
        f"Train:\n"
        f"| Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | "
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {train_f1:.4f}")

    # ---------------- 验证 ----------------
    model.eval()
    val_loss = 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for text, audio, vision, y in val_loader:
            text, audio, vision, y = text.to(DEVICE), audio.to(DEVICE), vision.to(DEVICE), y.to(DEVICE).long()
            t_proj = model.text_proj(text)
            a_proj = model.audio_proj(audio)
            v_proj = model.vision_proj(vision)

            # 验证阶段不使用 alpha
            fused = torch.cat([t_proj, a_proj, v_proj], dim=-1)
            fused = nn.functional.layer_norm(fused, fused.shape[-1:])
            output = model.perceiver(fused)

            val_loss += criterion(output, y).item() * y.size(0)
            val_preds.append(output.argmax(dim=1).cpu())
            val_labels.append(y.cpu())

    val_preds = torch.cat(val_preds)
    val_labels = torch.cat(val_labels)
    val_acc = accuracy_score(val_labels, val_preds)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        val_labels, val_preds, average='macro', zero_division=0
    )
    avg_val_loss = val_loss / len(val_loader.dataset)

    log(f"Validate:\n"
        f"| Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | "
        f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

    results.append({
        "epoch": epoch+1,
        "train_loss": avg_train_loss,
        "train_acc": train_acc,
        "train_precision": precision,
        "train_recall": recall,
        "train_f1": train_f1,
        "val_loss": avg_val_loss,
        "val_acc": val_acc,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1
    })

# ---------------- 保存 CSV ----------------
df = pd.DataFrame(results)
df.to_csv(csv_file, index=False)
log(f"Training finished. Results saved to {txt_file} and CSV.")

# ---------------- 测试阶段 ----------------
model.eval()
test_loss = 0
test_preds, test_labels = [], []
with torch.no_grad():
    for text, audio, vision, y in test_loader:
        text, audio, vision, y = text.to(DEVICE), audio.to(DEVICE), vision.to(DEVICE), y.to(DEVICE).long()
        t_proj = model.text_proj(text)
        a_proj = model.audio_proj(audio)
        v_proj = model.vision_proj(vision)

        fused = torch.cat([t_proj, a_proj, v_proj], dim=-1)
        fused = nn.functional.layer_norm(fused, fused.shape[-1:])
        output = model.perceiver(fused)

        test_loss += criterion(output, y).item() * y.size(0)
        test_preds.append(output.argmax(dim=1).cpu())
        test_labels.append(y.cpu())

test_preds = torch.cat(test_preds)
test_labels = torch.cat(test_labels)
test_acc = accuracy_score(test_labels, test_preds)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
    test_labels, test_preds, average='macro', zero_division=0
)
avg_test_loss = test_loss / len(test_loader.dataset)

log(f"Test:\n"
    f"| Loss: {avg_test_loss:.4f} | Acc: {test_acc:.4f} | "
    f"Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")
