import pickle
import numpy as np

# 你自己的pkl路径
pkl_path = "D:/MyProject/RobotPerceiver/data/open_dataset/mosi/Processed/aligned_50.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("\n🗂️  所有字段名：")
for k in data.keys():
    print(f"  - {k}")

print("\n📐 各字段结构预览：")
for k, v in data.items():
    if isinstance(v, (np.ndarray, list)):
        try:
            shape = v.shape if isinstance(v, np.ndarray) else (len(v),)
            print(f"  {k}: type={type(v).__name__}, shape={shape}")
        except Exception as e:
            print(f"  {k}: type={type(v).__name__}, shape=? (error: {e})")
    else:
        print(f"  {k}: type={type(v).__name__}")

# 选择一个 split 查看内容
split = 'test'
split_data = data[split]

print(f"\n📁 Split: {split}")
print("🔑 子字段名：", list(split_data.keys()))

for key, value in split_data.items():
    if isinstance(value, (list, np.ndarray)):
        print(f"  {key}: type={type(value).__name__}, len={len(value)}")
        if len(value) > 0 and hasattr(value[0], 'shape'):
            print(f"    单个样本 shape: {value[0].shape}")
        elif len(value) > 0 and isinstance(value[0], (list, np.ndarray)):
            print(f"    单个样本 shape: ({len(value[0])}, {len(value[0][0]) if isinstance(value[0][0], (list, np.ndarray)) else '?'})")
        else:
            print(f"    示例值: {value[0]}")
    else:
        print(f"  {key}: 非数组字段，类型为 {type(value).__name__}")