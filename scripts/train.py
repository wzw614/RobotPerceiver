import torch
from perceiver_pytorch import Perceiver
from utils.dataLoader import get_loader
from utils.config_loader import load_config
import torch.nn as nn
import torch.optim as optim

# 加载配置
cfg = load_config("./config/config.yaml")


class MultimodalPerceiver(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 投影层定义 (直接从YAML读取维度)
        self.text_proj = nn.Linear(cfg.modalities.text.feat_dim, cfg.model.latent_dim)
        self.audio_proj = nn.Linear(cfg.modalities.audio.feat_dim, cfg.model.latent_dim)
        self.vision_proj = nn.Linear(cfg.modalities.vision.feat_dim, cfg.model.latent_dim)

        # Perceiver核心
        self.perceiver = Perceiver(
            input_channels=cfg.model.latent_dim * 3,
            input_axis=1,  # 1D时序
            num_freq_bands=6,
            max_freq=10.,
            depth=cfg.model.depth,
            num_latents=cfg.model.num_latents,
            latent_dim=cfg.model.latent_dim,
            cross_heads=cfg.model.cross_heads,
            attn_dropout=cfg.model.dropout,
            num_classes=cfg.modalities.num_classes
        )

    def forward(self, text, audio, vision):
        text_feat = self.text_proj(text)  # [B,50,512]
        audio_feat = self.audio_proj(audio)  # [B,50,512]
        vision_feat = self.vision_proj(vision)  # [B,50,512]
        fused = torch.cat([text_feat, audio_feat, vision_feat], dim=-1)
        return self.perceiver(fused)


def evaluate(model, data_loader, criterion, device):
    # 分类任务
    criterion = nn.CrossEntropyLoss()

    """评估模型在验证集/测试集上的表现"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for text, audio, vision, labels in data_loader:
            inputs = {
                'text': text.to(device),
                'audio': audio.to(device),
                'vision': vision.to(device)
            }
            labels = labels.to(device)

            outputs = model(**inputs)
            loss = criterion(outputs, labels)# 使用传入的损失函数

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / total, correct / total

def train():
    # 分类任务
    criterion = nn.CrossEntropyLoss()

    # 数据加载 (使用YAML中的参数)
    train_loader = get_loader(
        cfg.data.path,
        "train",
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )
    val_loader = get_loader(cfg.data.path, "valid", cfg.data.batch_size)  # 新增
    test_loader = get_loader(cfg.data.path, "test", cfg.data.batch_size)  # 可选

    # 模型初始化
    model = MultimodalPerceiver(cfg).to(cfg.training.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = nn.CrossEntropyLoss()  # 新增损失函数

    # 训练循环
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for text, audio, vision, labels in train_loader:
            text = text.to(cfg.training.device)
            audio = audio.to(cfg.training.device)
            vision = vision.to(cfg.training.device)
            labels = labels.to(cfg.training.device)

            outputs = model(text, audio, vision)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 验证集评估
        val_loss, val_acc = evaluate(model, val_loader, criterion, cfg.training.device)

        # 打印日志
        print(f"Epoch {epoch + 1}/{cfg.training.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

    # 验证循环
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for text, audio, vision, labels in train_loader:
            # 数据转移到设备
            inputs = {
                'text': text.to(cfg.training.device),
                'audio': audio.to(cfg.training.device),
                'vision': vision.to(cfg.training.device)
            }
            labels = labels.to(cfg.training.device)

            # 前向传播
            outputs = model(**inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计训练指标
            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 计算验证集指标
        val_loss, val_acc = evaluate(model, val_loader, cfg.training.device)
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # 打印完整指标
        print(f"Epoch {epoch + 1}/{cfg.training.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")


if __name__ == "__main__":
    train()