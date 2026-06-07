# 在 imports 中添加（如果需要对比损失）
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity, kl_div

class Alignment(nn.Module):
    def __init__(self, method: str, **kwargs):
        super().__init__()
        self.method = method
        self.kwargs = kwargs  # e.g., {'temperature': 0.07} for contrastive

    def forward(self, z_self: torch.Tensor, z_ref: torch.Tensor) -> torch.Tensor:
        """
        z_self: 您的VAE潜在表示 (B, latent_dim, T)
        z_ref: 参考模型表示 (B, ref_dim, T) - 需预对齐维度
        返回: 对齐损失 (scalar per batch)
        """
        if self.method == "contrastive":  # InfoNCE风格
            # 假设已投影到相同维度；使用cosine sim
            sim = cosine_similarity(z_self.mean(2), z_ref.mean(2), dim=1) / self.kwargs.get('temperature', 0.07)
            # 简单正样本对齐（扩展到负样本需队列）
            loss = -torch.log_softmax(sim, dim=0).mean()  # 简化版；实际用NT-Xent
            return loss
        elif self.method == "mse":
            return F.mse_loss(z_self, z_ref)
        elif self.method == "kl":
            # 假设z_self/z_ref是分布参数 (mu, logvar)
            kl = kl_div(z_self, z_ref, reduction='batchmean')
            return kl
        else:
            raise ValueError(f"Unknown alignment method: {self.method}")
