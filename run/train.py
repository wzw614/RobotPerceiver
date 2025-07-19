import torch

"""
    训练
    定义训练一个 epoch 的过程
    model：自己定义的多模态情感识别模型，我的是perciever
    loader：训练数据的DataLoader
    optimizer：优化器 Adam
    device：cpu/Gpu
"""
def train_epoch(model, loader, optimizer, device):
    model.train()   # 设置模型为训练模式
    total_loss = 0.0    # 用来累计当前 epoch 中每个 batch 的损失值，以便最终求平均

    # 进入 mini-batch 循环，每次从 loader 中取出一小批样本
    for texts, audios, visions, labels in loader:
        optimizer.zero_grad()   # 清除上一步反向传播中累计的梯度，以免干扰本次梯度计算

        # 2. 模型前向传播，得到输出结果
        # 使用 .to(device) 将张量移动到 GPU 或 CPU
        outputs = model(texts.to(device), audios.to(device), visions.to(device))

        # 3. 计算损失
        # 使用交叉熵损失计算预测结果与真实标签的误差
        loss = torch.nn.CrossEntropyLoss()(outputs, labels.to(device))
        # 反向传播，计算每个参数的梯度
        loss.backward()

        # 4. 反向传播
        # 用计算好的梯度来更新模型参数
        optimizer.step()
        # 将当前 batch 的损失（用 .item() 取出数值）加到总损失中
        total_loss += loss.item()

    return {
        'loss': float(total_loss / len(loader)),    # 平均损失
        'batch_size': int(len(loader.dataset)),  # 训练集总样本数
        'learning_rate': float(optimizer.param_groups[0]['lr'])  # 当前学习率（方便后续学习率调度记录）
    }