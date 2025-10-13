from pathlib import Path
import numpy as np
from hsic_lasso import HSICLasso
from torch.utils.data import Dataset, DataLoader

# 保存路径
SAVE_DIR = Path("./alpha")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
ALPHA_PATH = SAVE_DIR / "alpha_masks.npz"

# 本来是没有想池化的，但是太大了，计算量过大，不好算，而且hsic阶段，token没那么重要，所以把token的维度池化了
def flatten_loader(loader: DataLoader):
    """
    将 train_loader 的全部数据池化 token 后展开，每个模态独立计算
    返回：
        feats_dict: {"text": [N, D], "audio": [N, D], "vision": [N, D]}
        labels: [N,]
    """
    text_list, audio_list, vision_list, labels_list = [], [], [], []

    for batch in loader:
        text, audio, vision, label = batch  # [B, T, D]
        B, T, D_text = text.shape
        _, _, D_audio = audio.shape
        _, _, D_vision = vision.shape

        # 1) 池化 token -> [B, D]
        text_pool = text.mean(dim=1).cpu().numpy()
        audio_pool = audio.mean(dim=1).cpu().numpy()
        vision_pool = vision.mean(dim=1).cpu().numpy()

        text_list.append(text_pool)
        audio_list.append(audio_pool)
        vision_list.append(vision_pool)

        # 标签不变
        labels_list.append(label.cpu().numpy())

    feats_dict = {
        "text": np.concatenate(text_list, axis=0),    # [N, D]
        "audio": np.concatenate(audio_list, axis=0),  # [N, D]
        "vision": np.concatenate(vision_list, axis=0) # [N, D]
    }
    labels = np.concatenate(labels_list, axis=0)  # [N,]
    return feats_dict, labels

def compute_alpha(x: np.ndarray, y: np.ndarray, topk: int | None = None) -> np.ndarray:
    model = HSICLasso()
    model.input(x, y)
    model.classification()  # 或 regression()

    scores = np.zeros(x.shape[1])
    selected = model.get_index()
    selected_scores = model.get_index_score()

    for i, idx in enumerate(selected):
        scores[idx] = selected_scores[i]

    if topk is not None:
        top_idx = np.argsort(scores)[-topk:]
        mask = np.zeros_like(scores)
        mask[top_idx] = scores[top_idx]
        scores = mask

    return scores

"""
    直接从 train_loader 计算 alpha（全部特征一次性计算）
"""
def complete_alpha(loader: DataLoader):

    print("Flattening train_loader and concatenating all features ...")
    feats_dict, y_all = flatten_loader(loader)

    alpha_dict = {}
    print("Computing HSIC-Lasso alpha for each modality ...")
    for modal, X in feats_dict.items():
        alpha = compute_alpha(X, y_all)
        alpha_dict[modal] = alpha
        print(f"Alpha {modal} shape: {alpha.shape}")

    # 保存为 npz
    np.savez(ALPHA_PATH, **alpha_dict)
    print(f"Alpha saved to {ALPHA_PATH.resolve()}")

    return alpha_dict

if __name__ == "__main__":
    print("开始计算alpha")