import matplotlib
# 必须在导入pyplot之前设置后端
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix


class ResultVisualizer:

    """
        初始化可视化工具
    """
    def __init__(self, cfg, logger):
        self.fig_dir = Path(cfg.output.fig_dir)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.train_loss = []
        self.val_acc = []

    """
        每个epoch更新可视化
    """
    def update(self, epoch, model, data_loader):
        # 1. 更新数据
        latest = self.logger.history['metrics'][-1]
        self.train_loss.append(latest['train']['loss'])
        self.val_acc.append(latest['val']['accuracy'])

        # 2. 绘制专业曲线图
        plt.figure(figsize=(12, 5))

        # 训练损失子图
        plt.subplot(121)
        plt.plot(self.train_loss, 'b-o', linewidth=2, markersize=4)
        plt.title(f"Training Loss (Epoch {epoch})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)

        # 验证准确率子图
        plt.subplot(122)
        plt.plot(self.val_acc, 'r-s', linewidth=2)
        plt.title(f"Validation Accuracy (Epoch {epoch})")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)  # 固定y轴范围
        plt.grid(True)

        # 保存和清理
        plt.tight_layout()
        plt.savefig(self.fig_dir / f"training_curve_epoch_{epoch}.png", dpi=300)
        plt.close()

    """
        生成最终报告
    """
    def generate_final_report(self, model, test_loader, device):
        print("正在生成最终报告...")

        # 1. 混淆矩阵
        from run.val import validate  # 避免循环导入
        test_metrics = validate(model, test_loader, device)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix(test_metrics['labels'], test_metrics['preds']),
            annot=True
        )
        plt.savefig(self.fig_dir / "confusion_matrix.png")
        plt.close()
        print("- 混淆矩阵已保存")

        # 2. 训练曲线
        self._plot_training_curve()
        print("- 训练曲线图已保存")

        # 3. 模态贡献度分析（示例）
        self._plot_modality_importance(model)
        print("- 模态重要性图已保存")

    """
        绘制训练损失和验证准确率曲线，支持两种模式：
            - epoch指定时：保存单epoch图片
            - epoch为None时：保存完整训练曲线（用于最终报告）
    """
    def _plot_training_curve(self, epoch=None):

        plt.figure(figsize=(12, 5))

        # 训练损失子图
        plt.subplot(121)
        x_values = range(len(self.train_loss))
        plt.plot(x_values, self.train_loss, 'b-o', label='Train Loss')
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if epoch is not None:
            plt.axvline(x=epoch, color='r', linestyle='--', alpha=0.3)

        # 验证准确率子图
        plt.subplot(122)
        plt.plot(x_values, self.val_acc, 'g-s', label='Val Accuracy')
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.05)
        if epoch is not None:
            plt.axvline(x=epoch, color='r', linestyle='--', alpha=0.3)

        plt.tight_layout()

        # 根据模式选择保存路径
        if epoch is not None:
            save_path = self.fig_dir / f"epoch_{epoch}.png"
        else:
            save_path = self.fig_dir / "final_training_curve.png"

        plt.savefig(save_path, dpi=300)
        plt.close()

    """
    绘制模态重要性
    """
    def _plot_modality_importance(self, model):
        importance = model.get_modality_importance()
        plt.bar(['Text', 'Audio', 'Vision'], importance)
        plt.savefig(self.fig_dir / "modality_importance.png")
        plt.close()