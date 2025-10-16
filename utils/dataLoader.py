
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.pkl_utils import load_split
from utils.config_loader import load_config

# ---------------- 读取配置 ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
cfg = load_config(BASE_DIR / "config/config.yaml")
MAP_DIM = cfg.model.latent_dim  # 从配置文件读取 要映射到的维度

class MOSIDataset(Dataset):

    def __init__(self, pkl_path: str | Path, split: str = "train", device: str | torch.device = "cpu"):
        super().__init__()
        d = load_split(pkl_path, split) # 加载原始数据

        # 1. 原始特征转 tensor
        self.text = torch.as_tensor(d["text"], dtype=torch.float32, device=device)       # [N, 50, 768]
        self.audio = torch.as_tensor(d["audio"], dtype=torch.float32, device=device)     # [N, 50,   5]
        self.vision = torch.as_tensor(d["vision"], dtype=torch.float32, device=device)   # [N, 50,  20]

        # 2. 标签
        self.label = torch.as_tensor(d["classification_labels"], dtype=torch.long, device=device)

        print(self.text.shape)  # [B, T, feature_dim]
        print(self.audio.shape)
        print(self.vision.shape)

        # 3. 映射到latent相同维度
        self.linear_map(MAP_DIM)

    def linear_map(self, map_dim: int):
        """用 nn.Linear 将三模态统一映射到 latent_dim，映射后固定不再训练"""
        # 定义线性层（只用一次，不训练）
        text_proj = nn.Linear(self.text.size(-1), map_dim, bias=False).to('cpu')
        audio_proj = nn.Linear(self.audio.size(-1), map_dim, bias=False).to('cpu')
        vision_proj = nn.Linear(self.vision.size(-1), map_dim, bias=False).to('cpu')

        # 映射整个 dataset
        with torch.no_grad():  # 不需要梯度
            self.text = text_proj(self.text)
            self.audio = audio_proj(self.audio)
            self.vision = vision_proj(self.vision)
        print(f"映射后text的维度为:{self.text.shape}")
        print(f"映射后audio的维度为:{self.audio.shape}")
        print(f"映射后vision的维度为:{self.vision.shape}")
    def __len__(self) -> int:
        return self.text.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.text[idx],      # [50, 768]
            self.audio[idx],     # [50,   5]
            self.vision[idx],    # [50,  20]
            self.label[idx],     # scalar
        )

"""
    加载多模态数据
"""
def get_loader(pkl_path: str | Path,
               split: str,
               batch_size: int = 16,
               shuffle: bool = True,
               num_workers: int = 0,
               pin_memory: bool = True) -> DataLoader:
    ds = MOSIDataset(pkl_path, split)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle if split == "train" else False,   # 仅训练集打乱顺序
                      num_workers=num_workers,  # 建议值为CPU核心数的50-75%
                      pin_memory=pin_memory)    # pin_memory=True，加速训练时从CPU到GPU传输
