"""
    仅做数据格式转换
    不做任何特征变换


    Dataset / DataLoader 封装。只做：
    1) 调 utils.load_split 读 dict
    2) 把三模态 + label 转成 tensor
   （保持原始维度；线性映射、池化等留给模型）
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from utils.pkl_utils import load_split


class MOSIDataset(Dataset):

    def __init__(self, pkl_path: str | Path, split: str = "train", device: str | torch.device = "cpu"):
        super().__init__()
        d = load_split(pkl_path, split) # 加载原始数据

        # 1） 转 tensor，返回self(text,autio,vision,label)元组
        self.text = torch.as_tensor(d["text"], dtype=torch.float32, device=device)       # [N, 50, 768]
        self.audio = torch.as_tensor(d["audio"], dtype=torch.float32, device=device)     # [N, 50,   5]
        self.vision = torch.as_tensor(d["vision"], dtype=torch.float32, device=device)   # [N, 50,  20]

        # 2） 标签（示例用 classification_labels；若做回归换字段）
        self.label = torch.as_tensor(d["classification_labels"], dtype=torch.long, device=device)

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
               batch_size: int = 32,
               shuffle: bool = True,
               num_workers: int = 0,
               pin_memory: bool = True) -> DataLoader:
    ds = MOSIDataset(pkl_path, split)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle if split == "train" else False,   # 仅训练集打乱顺序
                      num_workers=num_workers,  # # 建议值为CPU核心数的50-75%
                      pin_memory=pin_memory)    # pin_memory=True，加速训练时从CPU到GPU传输
