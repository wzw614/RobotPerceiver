# # scripts/hsic_lasso.py
# """
#     用 pyhsiclasso 计算 HSIC-Lasso token-level mask，保存为 .npy 文件。
#     依赖：pip install pyhsiclasso torch numpy
# """
#
# from pathlib import Path
# import numpy as np
# import torch
# import torch.nn as nn
# from pyHSICLasso import HSICLasso
# from utils.dataLoader import get_loader
#
# # ----- 配置参数 -----
# PKL_PATH = Path("D:/MyProject/RobotPerceiver/data/open_dataset/mosi/Processed/aligned_50.pkl")
# SPLIT = "train"
# BATCH_SIZE = 128
# PROJ_DIM = 256  # 统一映射到的维度
# SAVE_DIR = Path("./data/alpha")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)
# DEVICE = "cuda"
#
# SEQ_LEN = 50
#
# # ----- 打印工具 -----
# def preview(name, arr):
#     if isinstance(arr, torch.Tensor):
#         arr = arr.cpu().numpy()
#     print(f"{name:>12}: shape {arr.shape}")
#     print(arr[:2], "\n")       # 前 2 行
#
# # ----- 计算alpha -----
# def compute_alpha(X, y, name="Feature"):
#     """
#         使用 HSIC-Lasso 计算特征 alpha 权重
#         Args:
#             X: 特征矩阵 (n_samples, n_features)
#             y: 标签向量 (n_samples,)
#             name: 模态名称（仅用于打印日志）
#         Returns:
#             alpha: 特征重要性分数（与 X 的特征维度对应）
#     """
#     model = HSICLasso()
#     model.input(X, y)
#     model.classification()   # 或 regression()，取决于任务
#
#     # 得分结果，全部保留
#     scores = model.get_index_score()
#     idx = model.get_index()  # 所有特征按重要性排序
#
#     alpha = np.zeros(X.shape[1], dtype=np.float32)
#     for i, feature_idx in enumerate(idx):
#         alpha[feature_idx] = scores[i]
#
#     print(f"[HSIC-Lasso] {name} -> computed alpha for all features")
#     return alpha
#
# # ----- 主函数 -----
# def run():
#     loader = get_loader(PKL_PATH, SPLIT, batch_size=BATCH_SIZE, shuffle=False)
#
#     # 定义简单线性层做统一映射（不开训练），通通映射到256维
#     proj_text = nn.Linear(768, PROJ_DIM).to(DEVICE).eval()
#     proj_audio = nn.Linear(5, PROJ_DIM).to(DEVICE).eval()
#     proj_vision = nn.Linear(20, PROJ_DIM).to(DEVICE).eval()
#
#     text_pool, audio_pool, vision_pool, labels_all = [], [], [], []
#
#     # --------- 前向 + 池化 ----------
#     with torch.no_grad():
#         for text_seq, audio_seq, vision_seq, label in loader:
#             text_seq = text_seq.to(DEVICE)
#             audio_seq = audio_seq.to(DEVICE)
#             vision_seq = vision_seq.to(DEVICE)
#
#             t_emb = proj_text(text_seq).mean(dim=1).cpu()
#             a_emb = proj_audio(audio_seq).mean(dim=1).cpu()
#             v_emb = proj_vision(vision_seq).mean(dim=1).cpu()
#
#             text_pool.append(t_emb)
#             audio_pool.append(a_emb)
#             vision_pool.append(v_emb)
#             labels_all.append(label)
#
#     X_text = torch.cat(text_pool).numpy()
#     X_audio = torch.cat(audio_pool).numpy()
#     X_vision = torch.cat(vision_pool).numpy()
#     y_all = torch.cat(labels_all).numpy()
#
#     # --------- 计算 alpha ----------
#     alpha_text = compute_alpha(X_text, y_all, "TEXT")
#     alpha_audio = compute_alpha(X_audio, y_all, "AUDIO")
#     alpha_vision = compute_alpha(X_vision, y_all, "VISION")
#
#     # --------- 保存 ----------
#     np.save(SAVE_DIR / "alpha_text.npy", alpha_text)
#     np.save(SAVE_DIR / "alpha_audio.npy", alpha_audio)
#     np.save(SAVE_DIR / "alpha_vision.npy", alpha_vision)
#     print(f"\nα 已保存到: {SAVE_DIR.resolve()}")
#
# if __name__ == "__main__":
#     run()
#
#     alpha_text = np.load(SAVE_DIR / "alpha_text.npy")
#     alpha_audio = np.load(SAVE_DIR / "alpha_audio.npy")
#     alpha_vision = np.load(SAVE_DIR / "alpha_vision.npy")
#
#     print("alpha_text:", alpha_text.shape)
#     print("alpha_audio:", alpha_audio.shape)
#     print("alpha_vision:", alpha_vision.shape)
# scripts/compute_alpha_mask.py
import numpy as np
import torch
from pathlib import Path
from pyHSICLasso import HSICLasso
from utils.dataLoader import get_loader

# ---------------- 配置 ----------------
PKL_PATH = Path("D:/MyProject/RobotPerceiver/data/open_dataset/mosi/Processed/aligned_50.pkl")
SAVE_DIR = Path("./data/alpha")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
BATCH_SIZE = 16
DEVICE = "cuda"

PROJ_DIM = 256  # latent_dim，和 MultimodalPerceiver 保持一致

# ---------------- 简单投影函数 ----------------
def project_features(loader, proj_layer):
    """
    将原始特征投影到 latent_dim 并拼接 batch
    """
    proj_layer.eval()
    all_proj = []

    with torch.no_grad():
        for text, audio, vision, y in loader:
            x = text.to(DEVICE)  # 这里只是示例，如果是audio/vision换成相应
            x_proj = proj_layer(x)
            all_proj.append(x_proj.cpu().numpy().reshape(-1, x_proj.shape[-1]))  # [B*T, latent_dim]

    return np.concatenate(all_proj, axis=0)  # [N_tokens, latent_dim]

# ---------------- 计算 alpha ----------------
def compute_alpha(X, y, topk=None):
    """
    X: [N, D] 特征矩阵（N样本数，D特征维度）
    y: [N,] 标签
    返回 alpha: [D,] 每个特征维度的重要性
    """
    model = HSICLasso()
    model.input(X, y)
    model.classification()  # 或 regression()

    scores = np.zeros(X.shape[1])
    selected = model.get_index()
    selected_scores = model.get_index_score()

    for i, idx in enumerate(selected):
        scores[idx] = selected_scores[i]

    if topk is not None:
        # 保留 topk 特征，其余置0
        top_idx = np.argsort(scores)[-topk:]
        mask = np.zeros_like(scores)
        mask[top_idx] = scores[top_idx]
        scores = mask

    return scores

# ---------------- 主函数 ----------------
def run_alpha(cfg, model_projs):
    loader = get_loader(PKL_PATH, "train", batch_size=BATCH_SIZE, shuffle=False)

    # 准备标签
    all_labels = []
    for _, _, _, y in loader:
        all_labels.append(y.cpu().numpy().reshape(-1))
    y_all = np.concatenate(all_labels, axis=0)  # [N_tokens,]

    alpha_dict = {}
    for modal, proj in model_projs.items():
        print(f"Computing HSIC-Lasso for {modal} ...")
        X = project_features(loader, proj)  # [N_tokens, latent_dim]
        alpha = compute_alpha(X, y_all)
        alpha_dict[modal] = alpha
        print(f"Alpha {modal} shape: {alpha.shape}")

    # 保存
    np.savez(SAVE_DIR / "alpha_masks.npz",
             text=alpha_dict["text"],
             audio=alpha_dict["audio"],
             vision=alpha_dict["vision"])
    print(f"Alpha masks saved to {SAVE_DIR.resolve()}")

    return alpha_dict

# ---------------- 使用示例 ----------------
if __name__ == "__main__":
    # 假设你已经初始化了 MultimodalPerceiver 的投影层
    from models.perceiver import MultimodalPerceiver
    from utils.config_loader import load_config
    cfg = load_config("./config/config.yaml")
    model = MultimodalPerceiver(cfg)

    model_projs = {
        "text": model.text_proj,
        "audio": model.audio_proj,
        "vision": model.vision_proj
    }

    run_alpha(cfg, model_projs)
