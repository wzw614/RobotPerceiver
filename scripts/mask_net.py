'''
    动态掩码生成
    MLP


'''
import torch
from torch import nn
class MaskNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出每个 token 的 mask logit
        )

    def forward(self, x):
        mask_logits = self.net(x)  # [b, tokens, 1]
        mask = torch.sigmoid(mask_logits).squeeze(-1)  # [b, tokens] 可微分
        return mask
