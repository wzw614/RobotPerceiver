# import torch
# import numpy as np
# from sklearn.metrics import confusion_matrix    # 从 scikit-learn 导入 confusion_matrix 函数，用于生成混淆矩阵（表示各类预测情况）
#
# """
#     验证
#     model: 已训练好的模型
#     loader: 验证集的数据加载器（DataLoader）
#     device: CPU 或 GPU 设备对象
# """
# def validate(model, loader, device):
#     # 设置模型为 评估模式，会关闭训练时特有的模块行为：
#     # 比如 Dropout 层会关闭（不再随机丢弃神经元）
#     # BatchNorm 使用训练时记录的统计值而不是当前 batch 的统计值
#     model.eval()
#     # 准备两个空列表，用于收集整个验证集的：
#     # all_preds: 所有模型预测的标签
#     # all_labels: 所有对应的真实标签
#     all_preds, all_labels = [], []
#
#     # 使用 PyTorch 的上下文管理器：
#     # 表示不需要反向传播和梯度计算（节省显存、提高速度）
#     # 在验证、测试阶段必须加这个
#     with torch.no_grad():
#         # 遍历验证集的每个 batch
#         for texts, audios, visions, labels in loader:
#             outputs = model(texts.to(device), audios.to(device), visions.to(device))
#             preds = torch.argmax(outputs, dim=1)    #  从 logits 中取最大值所在的索引作为预测标签，即从每一行（一个样本）中找到最大概率的类别编号
#             all_preds.extend(preds.cpu().numpy())   # 将预测标签转换为 CPU 上的 numpy 数组，并追加到 all_preds 列表中
#             all_labels.extend(labels.numpy())       # 将真实标签（原本就可能在 CPU）转为 numpy，并追加到 all_labels 列表中。
#
#     # 确保返回简单可序列化的数据结构
#     return {
#         'accuracy': float(np.mean(np.array(all_preds) == np.array(all_labels))),
#         'preds': [int(x) for x in all_preds],   # 将所有预测值转为基本整数类型（int），用于记录或保存
#         'labels': [int(x) for x in all_labels], # 将所有真实标签也转为 int 类型
#         'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(), # 用 scikit-learn 计算混淆矩阵，表示预测类别 vs 真实类别的对照表
#         'raw_outputs': None  # 占位项，表示“原始模型输出值（logits）”没有被保存。如果以后需要分析 logits，可以在此返回
#     }

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def valid_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_labels = []

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for texts, audios, visions, labels in loader:
            texts, audios, visions, labels = texts.to(device), audios.to(device), visions.to(device), labels.to(device)

            outputs = model(texts, audios, visions)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'loss': avg_loss,
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
