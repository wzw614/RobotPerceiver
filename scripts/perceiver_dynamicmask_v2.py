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
from models.perceiver import MultimodalPerceiver

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
ALPHA_PATH = Path("./alpha")    # 掩码保存路径

# 2. 数据加载与处理
train_loader = get_loader(PKL_PATH, "train", batch_size=BATCH_SIZE, shuffle=True)
val_loader   = get_loader(PKL_PATH, "valid", batch_size=BATCH_SIZE, shuffle=False)
test_loader  = get_loader(PKL_PATH, "test", batch_size=BATCH_SIZE, shuffle=False)

# 3. 静态掩码计算与保存
alpha = complete_alpha(train_loader)

# 4. 模型构建

# 5. 训练、验证、测试

# 6. 日志保存

# 7. 可视化
#