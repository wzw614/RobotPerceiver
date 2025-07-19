from typing import Union, Dict, Any
from pathlib import Path
import yaml

"""
    将嵌套字典转换为可通过 cfg.model.latent_dim 方式访问的对象
"""
class AttrDict(dict):
    """
        支持点号访问的嵌套字典
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self    # 通过将 __dict__ 指向自身，实现属性式访问

    """
        递归转换字典为AttrDict
    """
    @staticmethod
    def recursive_convert(d):
        # 嵌套字典——转换
        if isinstance(d, dict):
            return AttrDict({k: AttrDict.recursive_convert(v) for k, v in d.items()})
        # 列表——转换
        elif isinstance(d, list):
            return [AttrDict.recursive_convert(i) for i in d]
        # 不可变类型——直接返回非容器类型（如int, str）
        return d


def load_config(yaml_path: Union[str, Path]) -> AttrDict:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 类型转换逻辑
    def convert_types(d):
        for k, v in d.items():
            if isinstance(v, str):
                try:
                    d[k] = float(v) if '.' in v or 'e' in v.lower() else int(v)
                except ValueError:
                    pass    # 保持字符串不变
            elif isinstance(v, dict):
                convert_types(v)    # 递归处理嵌套字典
        return d

    return AttrDict.recursive_convert(convert_types(config))