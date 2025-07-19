# scripts/hsic_lasso.py
"""
用 pyhsiclasso 计算 HSIC-Lasso 权重 α，保存为 .npy 文件。
依赖：pip install pyhsiclasso torch numpy
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from pyHSICLasso import HSICLasso
from utils.dataLoader import get_loader

# ----- 配置参数 -----
PKL_PATH = Path("J:/Lab_experiment/WZW/dataset/CMU-MOSI/Processed/aligned_50.pkl")
SPLIT = "train"
BATCH_SIZE = 128
PROJ_DIM = 256  # 统一映射到的维度
SAVE_DIR = Path("./data/alpha")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"  # 如果有GPU可改成 "cuda"

# ----- 打印形状、前两行、前五列 ----
def preview(name,arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    print(f"{name:>12}: shape {arr.shape}")
    print(arr[:2, :5], "\n")       # 前 2 行 × 前 5 列

# def compute_alpha(X: np.ndarray, y: np.ndarray,tag: str) -> np.ndarray:
#     """计算特征权重并归一化。并打印权重信息"""
#     print(f"\n=== HSIC‑Lasso for {tag} ===")
#     preview("X_pool", X)# 打印池化后的X
#     preview("y", y.reshape(-1, 1))# 打印标签y
#
#     model = HSICLasso()
#     model.input(X, y)
#     model.classification()  # 分类任务，回归改成 model.regression()
#
#
#     weights = model.beta[:, -1].astype(float)  # shape = (256,)
#
#     # weights = np.array(model.get_features()).astype(float)# 求权重并转为数值
#
#     preview("raw_w：", weights.reshape(1, -1))
#     alpha = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
#     preview("alpha：", alpha.reshape(1, -1))
#     return alpha
def compute_alpha(X: np.ndarray, y: np.ndarray, tag: str, proj_dim=256, smooth_ratio=0.2) -> np.ndarray:
    """计算带动态平滑的特征权重 alpha，并打印权重信息。
    smooth_ratio 是非选中特征权重相对于选中特征均值的比例。
    """
    print(f"\n=== HSIC‑Lasso for {tag} ===")
    preview("X_pool", X)
    preview("y", y.reshape(-1, 1))

    model = HSICLasso()
    model.input(X, y)
    model.classification()

    # 获取选中特征索引和对应的权重（非归一化）
    selected_indices = model.get_index()
    selected_weights_raw = model.get_index_score()

    print(f"选中特征数量: {len(selected_indices)}")
    preview("selected_weights_raw", selected_weights_raw.reshape(1, -1))

    # 归一化选中特征权重到 [0,1]
    weights_norm = (selected_weights_raw - selected_weights_raw.min()) / (selected_weights_raw.max() - selected_weights_raw.min() + 1e-8)
    preview("selected_weights_norm", weights_norm.reshape(1, -1))

    # 计算平滑值（非选中特征权重），是选中特征均值的 smooth_ratio 倍
    smooth_val = weights_norm.mean() * smooth_ratio
    print(f"平滑权重 smooth_val: {smooth_val:.4f}")

    # 构建完整权重向量，先填充平滑值
    alpha_full = np.full(proj_dim, smooth_val, dtype=np.float32)

    # 把选中特征权重赋进去
    alpha_full[selected_indices] = weights_norm

    preview("alpha_full", alpha_full.reshape(1, -1))
    return alpha_full


def run():
    loader = get_loader(PKL_PATH, SPLIT, batch_size=BATCH_SIZE, shuffle=False)

    # 定义简单线性层做统一映射（不开训练），通通映射到256维
    proj_text = nn.Linear(768, PROJ_DIM).to(DEVICE).eval()
    proj_audio = nn.Linear(5, PROJ_DIM).to(DEVICE).eval()
    proj_vision = nn.Linear(20, PROJ_DIM).to(DEVICE).eval()

    # 保存池化结果
    text_pool, audio_pool, vision_pool, labels_all = [], [], [], []

    # --------- 前向 + pool ----------
    for text_seq, audio_seq, vision_seq, label in loader:
        text_seq = text_seq.to(DEVICE)
        audio_seq = audio_seq.to(DEVICE)
        vision_seq = vision_seq.to(DEVICE)

        t_emb = proj_text(text_seq)  # [B, 50, d]
        a_emb = proj_audio(audio_seq)
        v_emb = proj_vision(vision_seq)

        text_pool.append(t_emb.mean(1).cpu())
        audio_pool.append(a_emb.mean(1).cpu())
        vision_pool.append(v_emb.mean(1).cpu())
        labels_all.append(label)
    # detach:切断反向传播的计算图    cpu：从gpu拷贝会cpu   numpy：转为numpy数组
    X_text = torch.cat(text_pool).detach().cpu().numpy()  # [N, d]
    X_audio = torch.cat(audio_pool).detach().cpu().numpy()
    X_vision = torch.cat(vision_pool).detach().cpu().numpy()
    y_all = torch.cat(labels_all).detach().cpu().numpy()

    # --------- HSIC‑Lasso ----------
    # alpha_text = compute_alpha(X_text, y_all, "TEXT")
    # alpha_audio = compute_alpha(X_audio, y_all, "AUDIO")
    # alpha_vision = compute_alpha(X_vision, y_all, "VISION")
    alpha_text = compute_alpha(X_text, y_all, "TEXT", proj_dim=PROJ_DIM, smooth_ratio=0.2)
    alpha_audio = compute_alpha(X_audio, y_all, "AUDIO", proj_dim=PROJ_DIM, smooth_ratio=0.2)
    alpha_vision = compute_alpha(X_vision, y_all, "VISION", proj_dim=PROJ_DIM, smooth_ratio=0.2)

    # --------- 保存 ----------
    np.save(SAVE_DIR / "alpha_text.npy", alpha_text)
    np.save(SAVE_DIR / "alpha_audio.npy", alpha_audio)
    np.save(SAVE_DIR / "alpha_vision.npy", alpha_vision)
    print(f"\nα 已保存到: {SAVE_DIR.resolve()}")


if __name__ == "__main__":
    run()
