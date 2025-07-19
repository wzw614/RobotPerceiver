# utils/pkl_utils.py
"""
轻量 I/O 工具：只负责把 *.pkl 读成 Python dict。
也提供一个 CLI 入口做结构探查（python -m utils.pkl_utils data/mosi.pkl --split train）
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import pickle
import argparse
import pprint


def load_split(pkl_path: str | Path, split: str = "train") -> Dict[str, Any]:
    """读取 pkl 并返回指定 split 字典。"""
    pkl_path = Path(pkl_path)
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if split not in data:
        raise KeyError(f"{split=} 不存在，可选 {list(data.keys())}")
    return data[split]


def summarize_split(split_dict: Dict[str, Any], depth: int = 2) -> None:
    """打印 dict 结构，debug 用。"""
    pprint.pp(split_dict.keys())
    for k, v in split_dict.items():
        shape = getattr(v, "shape", f"len={len(v)}") if isinstance(v, (list, tuple)) else getattr(v, "shape", "?")
        print(f"{k:>25s}: {type(v).__name__}  {shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_path")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    args = parser.parse_args()

    split = load_split(args.pkl_path, args.split)
    summarize_split(split)
