"""MeanFlow wrapper for one-step latent generation via average velocity prediction."""

import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    return (stopgrad(w) * delta_sq).mean()


class MeanFlowWrapper:
    """Wraps a backbone model for MeanFlow training and one-step sampling.

    Training: optimizes the average velocity field using JVP-based derivatives.
    Inference: generates latents in a single forward pass (Eq. 6 in the paper).
    """

    def __init__(
        self,
        channels=1,
        image_size=126,
        flow_ratio=0.50,
        time_dist=('lognorm', -0.4, 1.0),
        jvp_api='autograd',
    ):
        self.channels = channels
        self.image_size = image_size
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        assert jvp_api in ('funtorch', 'autograd')
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        else:
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[1], self.time_dist[2]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])
        indices = np.random.permutation(batch_size)[:int(self.flow_ratio * batch_size)]
        r_np[indices] = t_np[indices]
        return torch.tensor(t_np, device=device), torch.tensor(r_np, device=device)

    def loss(self, model, x, c, spk=None):
        """Compute MeanFlow training loss with JVP-based derivative estimation."""
        B, device = x.shape[0], x.device
        t, r = self.sample_t_r(B, device)
        t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
        r_ = rearrange(r, "b -> b 1 1 1").detach().clone()
        e = torch.randn_like(x)
        z = (1 - t_) * x + t_ * e
        v = e - x

        def model_partial(z, t, r):
            return model.forward(z, t, c, spk=spk)

        jvp_args = (model_partial, (z, t, r), (v, torch.ones_like(t), torch.zeros_like(r)))
        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v - (t_ - r_) * dudt
        error = u - stopgrad(u_tgt)
        return adaptive_l2_loss(error), (stopgrad(error) ** 2).mean()

    @torch.no_grad()
    def sample(self, model, batch_size, c, feat_dim, spk=None, device='cuda'):
        """One-step sampling: z_gen = z_1 - f_theta(z_1, 0, 1, c)."""
        model.eval()
        z = torch.randn(batch_size, self.channels, self.image_size, feat_dim, device=device)
        t = torch.full((batch_size,), 1.0, device=device)
        t_ = rearrange(t, "b -> b 1 1 1")
        r_ = torch.zeros_like(t_)
        v = model.forward(z, t, c, spk=spk)
        return z - (t_ - r_) * v


def differentiable_sample(model, wrapper, batch_size, c, feat_dim, spk=None, device='cuda'):
    """Differentiable one-step sampling (gradients flow back to model parameters)."""
    z = torch.randn(batch_size, wrapper.channels, wrapper.image_size, feat_dim,
                     device=device, requires_grad=True)
    t = torch.full((batch_size,), 1.0, device=device)
    r = torch.full((batch_size,), 0.0, device=device)
    t_ = rearrange(t, "b -> b 1 1 1")
    r_ = rearrange(r, "b -> b 1 1 1")
    v = model.forward(z, t, c, spk=spk)
    return z - (t_ - r_) * v
