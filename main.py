# ==================== main.py ====================
import torch
import numpy as np
from pathlib import Path
import sys
import csv
from datetime import datetime
import random

# 调整 Python 搜索路径，保证自定义模块能被顺利 import
sys.path.append(str(Path(__file__).parent))

from utils.config_loader import load_config
from run.train import train_epoch
from run.val import valid_epoch
from run.test import test_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 1. 加载配置
    cfg = load_config("./config/config.yaml")
    device = torch.device(cfg.training.device)
    SEED = cfg.training.seed
    set_seed(SEED)

    # 2. 数据加载
    from utils.dataLoader import get_loader
    train_loader = get_loader(cfg.data.path, "train", cfg.data.batch_size)
    val_loader = get_loader(cfg.data.path, "valid", cfg.data.batch_size)
    test_loader = get_loader(cfg.data.path, "test", cfg.data.batch_size)

    # 3. 模型初始化
    from models.perceiver import MultimodalPerceiver
    model = MultimodalPerceiver(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    # 4. 创建日志文件夹
    log_root = Path(f"D:/MyProject/RobotPerceiver/log/perceiver-mosi/seed{SEED}")
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_file = log_root / f"training_log_{timestamp}.txt"
    csv_file = log_root / f"training_log_{timestamp}.csv"

    # 写入 CSV 表头
    with open(csv_file, mode="w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["Epoch", "Train_Loss", "Train_Acc", "Train_Precision", "Train_Recall", "Train_F1",
                         "Val_Loss", "Val_Acc", "Val_Precision", "Val_Recall", "Val_F1"])

    # 5. 训练 + 验证循环
    EPOCHS = cfg.training.epochs
    with open(txt_file, "w") as f_txt:
        for epoch in range(EPOCHS):
            # --- 训练 ---
            train_metrics = train_epoch(model, train_loader, optimizer, device)
            # train_metrics: dict 应包含 loss, acc, precision, recall, f1
            log_str = (f"Epoch {epoch+1}/{EPOCHS}\n"
                       f"Train:\n"
                       f"| Loss: {train_metrics['loss']:.4f} | "
                       f"Acc: {train_metrics['acc']:.4f} | "
                       f"Precision: {train_metrics['precision']:.4f} | "
                       f"Recall: {train_metrics['recall']:.4f} | "
                       f"F1: {train_metrics['f1']:.4f}")
            print(log_str)
            f_txt.write(log_str + "\n")

            # --- 验证 ---
            val_metrics = valid_epoch(model, val_loader, device)
            log_str_val = (f"Validate:\n"
                            f"| Loss: {val_metrics['loss']:.4f} | "
                           f"Acc: {val_metrics['acc']:.4f} | "
                           f"Precision: {val_metrics['precision']:.4f} | "
                           f"Recall: {val_metrics['recall']:.4f} | "
                           f"F1: {val_metrics['f1']:.4f}")
            print(log_str_val)
            f_txt.write(log_str_val + "\n")

            # 写入 CSV
            with open(csv_file, mode="a", newline="") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow([
                    epoch+1,
                    train_metrics['loss'],
                    train_metrics['acc'],
                    train_metrics['precision'],
                    train_metrics['recall'],
                    train_metrics['f1'],
                    val_metrics['loss'],
                    val_metrics['acc'],
                    val_metrics['precision'],
                    val_metrics['recall'],
                    val_metrics['f1']
                ])

    # 6. 测试阶段
    test_metrics = test_model(model, test_loader, device)
    log_test = ("Test Metrics | " +
                " | ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()]))
    print(log_test)
    with open(txt_file, "a") as f_txt:
        f_txt.write(log_test + "\n")

if __name__ == "__main__":
    main()
