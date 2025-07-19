import torch                # 引入 PyTorch 主库，负责张量运算与深度学习
import numpy as np          # NumPy，做数值/数组辅助运算（本文件未直接用，但常备用）
from pathlib import Path    # 更方便的跨平台路径处理
import sys                  # 访问解释器相关功能，这里用来操作 import 搜索路径

# 调整 Python 搜索路径，保证自定义模块能被顺利 import
sys.path.append(str(Path(__file__).parent))

"""
    导入工程内自定义工具 / 逻辑模块
"""
from utils.config_loader import load_config     # 读取 YAML/JSON 的配置文件
from utils.logger import TrainingLogger         # 训练日志记录器
from utils.visualizer import ResultVisualizer   # 结果可视化
from run.train import train_epoch               # epoch 的训练函数
from run.val import validate                    # 验证集评估函数
from run.test import test_model                 # 测试集评估函数

"""
=========   主入口   =========
"""
def main():
    # 1. 加载配置
    print("=== 初始化开始 ===")
    cfg = load_config("./config/config.yaml")   # 读取超参 & 路径配置
    print("配置加载完成")
    device = torch.device(cfg.training.device)  # 选择 CPU / CUDA / MPS

    # 2. 初始化需要用到的工具
    logger = TrainingLogger(cfg)                # 创建日志记录器
    visualizer = ResultVisualizer(cfg,logger)   # 创建可视化器，需 logger 协助
    print("日志、可视化工具初始化完成")

    # 3. 数据加载
    from utils.dataLoader import get_loader  # 延迟导入避免循环依赖

    train_loader = get_loader(cfg.data.path, "train", cfg.data.batch_size)
    val_loader = get_loader(cfg.data.path, "valid", cfg.data.batch_size)
    test_loader = get_loader(cfg.data.path, "test", cfg.data.batch_size)
    # 依次返回三大数据集的 PyTorch DataLoader


    # 4. 模型初始化
    from models.perceiver import MultimodalPerceiver
    model = MultimodalPerceiver(cfg).to(device)     # 构造多模态 Perceiver 并放到 device
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)        # Adam 优化器


    # 5. 训练循环
    print("=== 训练开始 ===")
    for epoch in range(cfg.training.epochs):
        # 1) 训练阶段
        print("=== 训练阶段 ===")
        print(f"Epoch {epoch + 1}/{cfg.training.epochs} 开始")
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        print(f"训练完成 - 损失: {train_metrics['loss']:.4f}")

        # 2) 验证阶段
        val_metrics = validate(model, val_loader, device)       # 在验证集前向推理
        print(f"验证完成 - 准确率: {val_metrics['accuracy']:.2%}")

        # 3) 记录日志
        logger.log(epoch, train_metrics, val_metrics)           # 把本 epoch 的指标写入日志

        # 4) 可视化更新
        visualizer.update(epoch, model, val_loader)             # 动态画图/存可视化样例


    # 6. 最终测试
    print("=== 测试阶段 ===")
    test_metrics = test_model(model, test_loader, device)       # 在测试集上评估最终指标
    print("测试完成:", test_metrics)

    # 7. 记录日志、可视化
    print("=== 记录日志ing ===")
    logger.log_test(test_metrics)                               # 把测试结果写日志

    print("=== 可视化ing ===")
    visualizer.generate_final_report(model, test_loader, device)    # 生成终版图

    print("=== 程序运行结束 ===")

"""
    Python 直接运行该脚本时的入口（被 import 时不会执行）
"""
if __name__ == "__main__":
    main()