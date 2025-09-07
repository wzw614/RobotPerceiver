import torch
from perceiver_pytorch import Perceiver

class MultimodalPerceiver(torch.nn.Module):
    def __init__(self, cfg, alpha_text=None, alpha_audio=None, alpha_vision=None):
        super().__init__()
        latent_dim = cfg.model.latent_dim

        # 投影层，将原始模态特征映射到 latent_dim
        self.text_proj = torch.nn.Linear(cfg.modalities.text.feat_dim, latent_dim)
        self.audio_proj = torch.nn.Linear(cfg.modalities.audio.feat_dim, latent_dim)
        self.vision_proj = torch.nn.Linear(cfg.modalities.vision.feat_dim, latent_dim)

        # Perceiver 核心
        self.perceiver = Perceiver(
            input_channels=latent_dim * 3,    # 融合后的维度
            input_axis=1,                     # 时序维度
            num_freq_bands=6,
            max_freq=10.,
            depth=cfg.model.depth,
            num_latents=cfg.model.num_latents,
            latent_dim=latent_dim,
            cross_heads=cfg.model.cross_heads,
            attn_dropout=cfg.model.dropout,
            num_classes=cfg.modalities.num_classes
        )

        # HSIC-Lasso alpha mask，可直接传入 np.array 或 torch.tensor
        self.alpha_text = alpha_text if alpha_text is None or isinstance(alpha_text, torch.Tensor) else torch.tensor(alpha_text, dtype=torch.float32)
        self.alpha_audio = alpha_audio if alpha_audio is None or isinstance(alpha_audio, torch.Tensor) else torch.tensor(alpha_audio, dtype=torch.float32)
        self.alpha_vision = alpha_vision if alpha_vision is None or isinstance(alpha_vision, torch.Tensor) else torch.tensor(alpha_vision, dtype=torch.float32)

    def forward(self, text, audio, vision):
        """
        text: [B, T_text, text_dim]
        audio: [B, T_audio, audio_dim]
        vision: [B, T_vision, vision_dim]
        """
        b = text.shape[0]

        # 投影到 latent_dim
        t_proj = self.text_proj(text)      # [B, T_text, latent_dim]
        a_proj = self.audio_proj(audio)    # [B, T_audio, latent_dim]
        v_proj = self.vision_proj(vision)  # [B, T_vision, latent_dim]

        # 应用 alpha mask
        if self.alpha_text is not None:
            mask = self.alpha_text.to(text.device)
            t_proj = t_proj * mask.unsqueeze(0).unsqueeze(1)  # [1,1,latent_dim] -> broadcast到[B,T,latent_dim]

        if self.alpha_audio is not None:
            mask = self.alpha_audio.to(audio.device)
            a_proj = a_proj * mask.unsqueeze(0).unsqueeze(1)

        if self.alpha_vision is not None:
            mask = self.alpha_vision.to(vision.device)
            v_proj = v_proj * mask.unsqueeze(0).unsqueeze(1)

        # 融合三模态
        fused = torch.cat([t_proj, a_proj, v_proj], dim=-1)  # [B, T_total, latent_dim*3]

        # Perceiver 前向
        return self.perceiver(fused)
