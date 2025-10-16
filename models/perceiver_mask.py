'''
修改原始model
在forward中初始化动态掩码并传递掩码
'''
import torch
from torch import nn
from perceiver_pytorch import Perceiver

class PerceiverMask(nn.Module):
    def __init__(self, cfg, alpha_text=None, alpha_audio=None, alpha_vision=None):
        super().__init__()
        latent_dim = cfg.model.latent_dim

        # Perceiver主体
        self.perceiver = Perceiver(
            input_channels=latent_dim*3,  # 融合后的维度
            input_axis=1,  # 时序维度
            num_freq_bands=6,
            max_freq=10.,
            depth=cfg.model.depth,
            num_latents=cfg.model.num_latents,
            latent_dim=latent_dim,
            cross_heads=cfg.model.cross_heads,
            attn_dropout=cfg.model.dropout,
            num_classes=cfg.modalities.num_classes,
            fourier_encode_data = False
        )

        # 拼接静态掩码
        self.static_mask = self._concat_alpha(alpha_text, alpha_audio, alpha_vision)
        print(f"模型初始化后，静态掩码拼接为：{self.static_mask.shape}") # torch.Size([768])

    def _to_tensor(self, x):
        if x is None:
            return None
        return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

    def _concat_alpha(self, *alphas):
        tensors = [self._to_tensor(a) for a in alphas if a is not None]
        if len(tensors) == 0:
            return None
        return torch.cat(tensors, dim=-1)

    def forward(self, text, audio, vision):
        B = text.size(0)
        print(f"调用模型——拼接前文本形状: {text.shape}")
        data = torch.cat([text, audio, vision], dim=-1)  # [B, T, 768]
        print(f"调用模型——拼接后数据形状为：{data.shape}")

        if self.static_mask is not None:
            static_mask = self.static_mask.unsqueeze(0).expand(B, -1).contiguous()
            print(f"调用模型——扩展后静态掩码形状为：{static_mask.shape}")
        else:
            static_mask = None
        return self.perceiver(data, static_mask=static_mask)
