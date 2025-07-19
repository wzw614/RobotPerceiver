# 导入标准库
import json
import numpy as np
from pathlib import Path
from datetime import datetime


class TrainingLogger:
    """
    初始化日志系统
    """
    def __init__(self, cfg):
        self.log_dir = Path(cfg.output.log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.history = {
            'config': self._convert_config(cfg),  # 转换配置中的NumPy类型
            'metrics': [],
            'test': None
        }

    """
    将配置中的NumPy类型转换为Python原生类型
    """
    def _convert_config(self, cfg):

        if isinstance(cfg, dict):
            return {k: self._convert_config(v) for k, v in cfg.items()}
        elif isinstance(cfg, (np.integer, np.floating)):
            return int(cfg) if isinstance(cfg, np.integer) else float(cfg)
        elif isinstance(cfg, (list, tuple)):
            return [self._convert_config(x) for x in cfg]
        return cfg

    """
    记录日志并自动转换NumPy类型
    """
    def log(self, epoch, train_metrics, val_metrics):

        print(f"\nEpoch {epoch} 结果:")
        print(f"  训练损失: {train_metrics['loss']:.4f}")
        print(f"  验证准确率: {val_metrics['accuracy']:.2%}")

        entry = {
            'epoch': int(epoch),  # 明确转换为Python int
            'train': self._convert_metrics(train_metrics),
            'val': self._convert_metrics(val_metrics),
            'timestamp': str(datetime.now())
        }
        self.history['metrics'].append(entry)
        self._save()

    """
    转换指标中的NumPy类型
    """
    def _convert_metrics(self, metrics):
        converted = {}
        for k, v in metrics.items():
            try:
                if v is None:
                    converted[k] = None
                elif isinstance(v, (np.integer, int)):
                    converted[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    converted[k] = float(v)
                elif isinstance(v, (list, np.ndarray)):
                    # 处理嵌套列表的情况
                    if len(v) > 0 and isinstance(v[0], (list, np.ndarray)):
                        converted[k] = [[float(y) for y in x] for x in v]
                    else:
                        converted[k] = [float(x) for x in v]
                elif isinstance(v, dict):
                    converted[k] = self._convert_metrics(v)  # 递归处理字典
                else:
                    converted[k] = str(v)  # 最后尝试转为字符串
            except Exception as e:
                print(f"警告: 转换指标 {k} 时出错: {str(e)}")
                converted[k] = str(v)
        return converted

    def log_test(self, test_metrics):
        """记录测试日志"""
        self.history['test'] = test_metrics
        self._save()
        print(f"\n[TEST] Accuracy: {test_metrics['accuracy']:.2%}")

    """
    保存到文件
    """
    def _save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=self._json_default)

    """
    JSON序列化备用方法
    """
    def _json_default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return int(obj) if isinstance(obj, np.integer) else float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")