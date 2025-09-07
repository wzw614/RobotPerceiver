import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def test_model(model, loader, device):
    """
    在测试集上评估模型
    model: 多模态Perceiver模型
    loader: 测试集DataLoader
    device: cpu / cuda
    """
    model.eval()  # 设置为评估模式，不计算梯度

    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for texts, audios, visions, labels in loader:
            texts, audios, visions, labels = texts.to(device), audios.to(device), visions.to(device), labels.to(device)
            outputs = model(texts, audios, visions)

            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 预测
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_loss = total_loss / len(loader)

    # 打印日志（和训练验证保持一致）
    print(
        f"Test Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    return {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_samples': len(loader.dataset)
    }
