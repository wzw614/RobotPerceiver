import torch
from perceiver_pytorch import Perceiver

class MultimodalPerceiver(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # BERT→潜在空间
        # 确保 cfg.model.latent_dim ≥ 最大输入维度的1/3（此处768/3=256）
        self.text_proj = torch.nn.Linear(cfg.modalities.text.feat_dim, cfg.model.latent_dim)
        # 音频→潜在空间
        self.audio_proj = torch.nn.Linear(cfg.modalities.audio.feat_dim, cfg.model.latent_dim)
        # 视觉→潜在空间
        self.vision_proj = torch.nn.Linear(cfg.modalities.vision.feat_dim, cfg.model.latent_dim)

        # perceiver核心配置
        self.perceiver = Perceiver(
            input_channels=cfg.model.latent_dim * 3,    # 融合后维度,256*3
            input_axis=1,                               # 理时序维度（50帧）
            num_freq_bands=6,                           # 位置编码频带数
            max_freq=10.,                               # 最高频率
            depth=cfg.model.depth,                      # 与config.yaml一致
            num_latents=cfg.model.num_latents,          # 潜在向量数量
            latent_dim=cfg.model.latent_dim,            # 潜在向量维度
            cross_heads=cfg.model.cross_heads,          # 跨注意力头数
            attn_dropout=cfg.model.dropout,             # 注意力dropout
            num_classes=cfg.modalities.num_classes
        )

        self.last_train_loss = None  # 添加训练损失记录
        self.last_val_acc = None  # 添加验证准确率记录

    """
        计算各模态特征的相对重要性
    """
    def get_modality_importance(self):
        # 示例实现：使用投影层的权重范数作为重要性指标
        return [
            torch.norm(self.text_proj.weight).item(),
            torch.norm(self.audio_proj.weight).item(),
            torch.norm(self.vision_proj.weight).item()
        ]

    # 前向传播
    def forward(self, text, audio, vision):
        text_feat = self.text_proj(text)    # 把文本特征投影到共享潜在维度
        audio_feat = self.audio_proj(audio)
        vision_feat = self.vision_proj(vision)
        fused = torch.cat([text_feat, audio_feat, vision_feat], dim=-1) # 将三个模态在最后一个维度上拼接，得到融合特征，dim为-1，表示特征维度上的拼接
        return self.perceiver(fused)    # 把融合后的特征送入 Perceiver 主干网络并输出结果。