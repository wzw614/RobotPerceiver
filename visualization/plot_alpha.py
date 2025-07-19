# scripts/plot_alpha_importance.py
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# 配置路径
ALPHA_DIR = Path("J:/Lab_experiment/WZW/RobotPerceiver/data/alpha")   # 之前保存alpha的文件夹
RESULTS_BAR_DIR = Path("J:/Lab_experiment/WZW/RobotPerceiver/results/alpha_importance_bars")
RESULTS_HEATMAP_DIR = Path("J:/Lab_experiment/WZW/RobotPerceiver/results/alpha_heatmaps")
RESULTS_3D_BAR_DIR = Path("J:/Lab_experiment/WZW/RobotPerceiver/results/alpha_3d_bars")
RESULTS_HIST_DIR = Path("J:/Lab_experiment/WZW/RobotPerceiver/results/alpha_histograms")


RESULTS_BAR_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_3D_BAR_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_HIST_DIR.mkdir(parents=True, exist_ok=True)

def plot_bar(alpha, modality_name):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(alpha)), alpha, color='skyblue')
    plt.title(f"{modality_name} Feature Importance (HSIC-Lasso α)")
    plt.xlabel("Feature Index")
    plt.ylabel("Normalized Importance")
    plt.tight_layout()
    save_path = RESULTS_BAR_DIR / f"{modality_name.lower()}_bar.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved bar plot to {save_path}")

def plot_heatmap(alpha, modality_name):
    plt.figure(figsize=(12, 1.5))
    sns.heatmap(alpha[np.newaxis, :], cmap="plasma", cbar=True, xticklabels=False, yticklabels=False)
    plt.title(f"{modality_name} Feature Importance Heatmap")
    plt.tight_layout()
    save_path = RESULTS_HEATMAP_DIR / f"{modality_name.lower()}_heatmap.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved heatmap to {save_path}")

def plot_3d_bar(alphas_dict):
    """
    绘制三模态特征重要性3D柱状图，添加渐变色。
    """
    import matplotlib.colors as mcolors
    from matplotlib import cm

    modalities = list(alphas_dict.keys())  # ['TEXT', 'AUDIO', 'VISION']
    num_modalities = len(modalities)
    num_features = len(next(iter(alphas_dict.values())))

    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(111, projection='3d')

    _x = np.arange(num_features)
    _y = np.arange(num_modalities)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    # 模态颜色基准（红、黄、蓝）
    base_colors = {
        'TEXT': cm.Reds,
        'AUDIO': cm.Greens,
        'VISION': cm.Blues
    }

    # 每个柱子的高度
    top = np.array([alphas_dict[modalities[y_i]][x_i] for x_i, y_i in zip(x, y)])
    bottom = np.zeros_like(top)
    width = depth = 0.8

    # 根据不同模态使用不同colormap，并按高度归一化映射颜色
    norm = plt.Normalize(vmin=0, vmax=top.max())
    colors = [base_colors[modalities[y_i]](norm(top_i)) for top_i, y_i in zip(top, y)]

    ax.bar3d(x, y, bottom, width, depth, top, shade=True, color=colors)

    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Modality")
    ax.set_zlabel("Weight (α)")
    ax.set_yticks(np.arange(num_modalities))
    ax.set_yticklabels(modalities)
    ax.set_title("3D Bar Chart of Feature Importance Across Modalities (Gradient)")

    save_path = RESULTS_3D_BAR_DIR / "alpha_3d_bar_gradient.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved gradient 3D bar plot to {save_path}")

def plot_histogram(alpha, modality_name, save_dir: Path):
    plt.figure(figsize=(6,4))
    n, bins, patches = plt.hist(alpha, bins=30, color='steelblue', edgecolor='black')
    plt.title(f'{modality_name} Feature Weight Distribution')
    plt.xlabel('Feature Weight (α)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)

    # 给每个柱子加上数值
    for count, patch in zip(n, patches):
        height = patch.get_height()
        if height > 0:  # 只显示非零的柱子数值
            plt.text(patch.get_x() + patch.get_width() / 2, height, f'{int(count)}',
                     ha='center', va='bottom', fontsize=8, rotation=0)

    plt.tight_layout()
    save_path = save_dir / f'{modality_name.lower()}_weight_hist.png'
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] {modality_name} weight histogram saved to {save_path}")


def main():
    alphas = {}
    for modality in ["text", "audio", "vision"]:
        alpha_path = ALPHA_DIR / f"alpha_{modality}.npy"
        alpha = np.load(alpha_path)
        print(f"{modality} alpha shape:", alpha.shape)

        plot_bar(alpha, modality.upper())
        plot_heatmap(alpha, modality.upper())
        alphas[modality.upper()] = alpha
        plot_histogram(alpha, modality.upper(), RESULTS_HIST_DIR)

    plot_3d_bar(alphas)


if __name__ == "__main__":
    main()
