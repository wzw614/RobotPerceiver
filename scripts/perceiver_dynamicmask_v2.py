'''
重写的 main 文件
功能：
    1. 初始化配置与环境
    2. 数据加载与处理
    3. 静态掩码计算与保存
    4. 模型构建
    5. 训练、验证、测试
    6. 日志保存
    7. 可视化
'''

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

from scripts.hsic_lasso_v2 import complete_alpha
from utils.config_loader import load_config
from utils.dataLoader import get_loader
from models.perceiver_mask import PerceiverMask

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1. 初始化配置与环境
BASE_DIR = Path(__file__).resolve().parent.parent   # 项目根目录
cfg = load_config(BASE_DIR / "config/config.yaml")  # 配置文件
DEVICE = torch.device(cfg.training.device)          # 设备类型
SEED = cfg.training.seed                            # 设置随机种子
set_seed(SEED)

BATCH_SIZE = cfg.data.batch_size                    # batch_size
EPOCHS = cfg.training.epochs                        # 训练轮数
LR = cfg.training.lr                                # 学习率

PKL_PATH = Path("D:/MyProject/RobotPerceiver/data/open_dataset/mosi/Processed/aligned_50.pkl")  # 数据路径
ALPHA_PATH = Path("./alpha/alpha_masks.npz")    # 掩码保存路径

# 2. 数据加载与处理
train_loader = get_loader(PKL_PATH, "train", batch_size=BATCH_SIZE, shuffle=True)
val_loader   = get_loader(PKL_PATH, "valid", batch_size=BATCH_SIZE, shuffle=False)
test_loader  = get_loader(PKL_PATH, "test", batch_size=BATCH_SIZE, shuffle=False)

# 3. 静态掩码计算与保存
if ALPHA_PATH.exists():
    print(f"***** 已检测到 alpha 文件：{ALPHA_PATH} *****")
    data = np.load(ALPHA_PATH)
    alpha_text = data["text"]
    alpha_audio = data["audio"]
    alpha_vision = data["vision"]
    print("***** Alpha 文件加载完成 *****")
else:
    print("***** 未检测到 alpha 文件，开始计算 HSIC-Lasso 掩码 *****")
    alpha_dict = complete_alpha(train_loader)
    alpha_text = alpha_dict["text"]
    alpha_audio = alpha_dict["audio"]
    alpha_vision = alpha_dict["vision"]
    print("***** Alpha 计算完成并保存 *****")

print(f"alpha_text的维度为:{alpha_text.shape}")
print(f"alpha_audio的维度为:{alpha_audio.shape}")
print(f"alpha_vision的维度为:{alpha_vision.shape}")

# 4. 模型构建
model = PerceiverMask(cfg, alpha_text=alpha_text, alpha_audio=alpha_audio, alpha_vision=alpha_vision).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# 5. 训练、验证、测试
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    all_preds, all_labels = [], []

    for text, audio, vision, y in train_loader:
        text,audio,vision,y = text.to(DEVICE),audio.to(DEVICE),vision.to(DEVICE),y.to(DEVICE).long()
        print(f"训练脚本——text形状为：{text.shape}")
        print(f"训练脚本——audio形状为：{audio.shape}")
        print(f"训练脚本——vision形状为：{vision.shape}")

        output = model(text,audio,vision)
        optimizer.zero_grad()

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



# 6. 日志保存

# 7. 可视化
#