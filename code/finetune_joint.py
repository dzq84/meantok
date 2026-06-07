"""End-to-end joint refinement: trainable MeanFlow (low LR) + trainable VAE decoder.

Key difference from decoder-only refinement:
  - MeanFlow is NOT frozen; gradients flow through the one-step sampling.
  - Uses differentiable_sample() so waveform-domain losses update both the
    latent generator and the decoder jointly.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import json
import argparse
import logging
import math
from pathlib import Path
from typing import Optional, Dict, Any

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.autoencoders import AudioAutoencoder
from stable_audio_tools.models.discriminators import EncodecDiscriminator, get_hinge_losses
from stable_audio_tools.training.losses import MultiLoss, ValueLoss, AuralossLoss, auraloss
from stable_audio_tools.training.autoencoders import trim_to_shortest

from speechflow.modules.dit1d import DiT1D
from speechflow.flow.meanflow import MeanFlowWrapper, differentiable_sample
from paired_dataloader import get_paired_dataloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class JointFinetuneModule(pl.LightningModule):
    """End-to-end joint refinement: trainable MeanFlow + trainable VAE decoder."""

    def __init__(
        self,
        meanflow_ckpt_path: str,
        autoencoder_config_path: str,
        vae_ckpt_path: str,
        lr: float = 1e-5,
        meanflow_lr: float = 1e-6,
        disc_lr: float = 3e-5,
        warmup_steps: int = 1000,
        use_discriminator: bool = True,
        adv_weight: float = 0.1,
        fm_weight: float = 5.0,
        mrstft_weight: float = 1.0,
        chunk_size: int = 126,
        spk_dim: int = 192,
        feat_dim: int = 16,
        dit_hidden_size: int = 1152,
        dit_depth: int = 28,
        dit_num_heads: int = 16,
        max_audio_samples: int = 24000 * 10,
        sample_rate: int = 24000,
        token_rate: int = 25,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.max_audio_samples = max_audio_samples
        self.lr = lr
        self.meanflow_lr = meanflow_lr
        self.disc_lr = disc_lr
        self.warmup_steps = warmup_steps
        self.use_discriminator = use_discriminator
        self.chunk_size = chunk_size
        self.feat_dim = feat_dim
        self.spk_dim = spk_dim
        self.sample_rate = sample_rate
        self.token_rate = token_rate
        self.samples_per_token = sample_rate // token_rate

        # Load MeanFlow (trainable)
        self.meanflow_model = DiT1D(
            input_size=chunk_size, patch_size=1, in_channels=1,
            hidden_size=dit_hidden_size, depth=dit_depth, num_heads=dit_num_heads,
            feat_dim=feat_dim, num_tokens=6561, spk_dim=spk_dim,
        )
        self.meanflow_wrapper = MeanFlowWrapper(
            channels=1, image_size=chunk_size, flow_ratio=0.50,
            time_dist=('lognorm', -0.4, 1.0), jvp_api='autograd',
        )
        ckpt = torch.load(meanflow_ckpt_path, map_location='cpu')
        sd = ckpt.get("state_dict", ckpt)
        mf_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        mf_sd = {k: v for k, v in mf_sd.items() if not k.endswith('pos_embed')}
        self.meanflow_model.load_state_dict(mf_sd, strict=False)

        # Load VAE (freeze encoder/bottleneck, keep decoder trainable)
        with open(autoencoder_config_path, 'r') as f:
            ae_config = json.load(f)
        self.autoencoder = create_model_from_config(ae_config)
        vae_ckpt = torch.load(vae_ckpt_path, map_location='cpu')
        vae_sd = vae_ckpt.get("state_dict", vae_ckpt)
        adj_sd = {(k[12:] if k.startswith("autoencoder.") else k): v for k, v in vae_sd.items()}
        adj_sd = {k: v for k, v in adj_sd.items() if not k.endswith('pos_embed')}
        self.autoencoder.load_state_dict(adj_sd, strict=False)
        for module in [self.autoencoder.encoder, self.autoencoder.bottleneck, self.autoencoder.pretransform]:
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False

        # Losses
        scales = [2048, 1024, 512, 256, 128, 64, 32]
        self.mrstft = auraloss.MultiResolutionSTFTLoss(
            sample_rate=sample_rate, fft_sizes=scales,
            hop_sizes=[s // 4 for s in scales], win_lengths=scales, perceptual_weighting=True,
        )
        self.discriminator = None
        if use_discriminator:
            self.discriminator = EncodecDiscriminator(
                in_channels=1, filters=64,
                n_ffts=scales[:5], hop_lengths=[s // 4 for s in scales[:5]], win_lengths=scales[:5],
            )
        self.gen_loss_modules = [
            AuralossLoss(self.mrstft, target_key='reals', input_key='decoded',
                         name='mrstft_loss', weight=mrstft_weight),
        ]
        if use_discriminator:
            self.gen_loss_modules += [
                ValueLoss(key='loss_adv', weight=adv_weight, name='loss_adv'),
                ValueLoss(key='feature_matching_distance', weight=fm_weight, name='feature_matching_loss'),
            ]
        self.losses_gen = MultiLoss(self.gen_loss_modules)
        if use_discriminator:
            self.losses_disc = MultiLoss([ValueLoss(key='loss_dis', weight=1.0, name='discriminator_loss')])
        self.warmed_up = False

    def configure_optimizers(self):
        opt_gen = torch.optim.AdamW([
            {"params": list(self.meanflow_model.parameters()), "lr": self.meanflow_lr},
            {"params": list(self.autoencoder.decoder.parameters()), "lr": self.lr},
        ], betas=(0.8, 0.99), weight_decay=1e-3)
        opts = [opt_gen]
        if self.use_discriminator:
            opts.append(torch.optim.AdamW(self.discriminator.parameters(), lr=self.disc_lr,
                                          betas=(0.8, 0.99), weight_decay=1e-3))
        return opts

    def generate_latent_from_meanflow(self, spkemb, tokens, token_lengths):
        """Differentiable chunk-based latent generation (gradients flow to MeanFlow)."""
        B, device = spkemb.shape[0], spkemb.device
        if spkemb.dim() == 2:
            spkemb = spkemb.unsqueeze(1)
        max_tl = token_lengths.max().item()
        chunks = []
        for ci in range((max_tl + self.chunk_size - 1) // self.chunk_size):
            s, e = ci * self.chunk_size, min((ci + 1) * self.chunk_size, max_tl)
            ct = tokens[:, s:e]
            if e - s < self.chunk_size:
                ct = torch.cat([ct, torch.zeros(B, self.chunk_size - (e - s), dtype=ct.dtype, device=device)], 1)
            xg = differentiable_sample(self.meanflow_model, self.meanflow_wrapper, B, ct,
                                        self.feat_dim, spk=spkemb, device=device)
            chunks.append(xg.permute(0, 1, 3, 2).squeeze(1)[:, :, :e - s])
        return torch.cat(chunks, dim=2), token_lengths.clone().to(device)

    @torch.no_grad()
    def generate_latent_from_meanflow_infer(self, spkemb, tokens, token_lengths):
        """Non-differentiable version for inference (saves memory)."""
        self.meanflow_model.eval()
        B, device = spkemb.shape[0], spkemb.device
        if spkemb.dim() == 2:
            spkemb = spkemb.unsqueeze(1)
        max_tl = token_lengths.max().item()
        chunks = []
        for ci in range((max_tl + self.chunk_size - 1) // self.chunk_size):
            s, e = ci * self.chunk_size, min((ci + 1) * self.chunk_size, max_tl)
            ct = tokens[:, s:e]
            if e - s < self.chunk_size:
                ct = torch.cat([ct, torch.zeros(B, self.chunk_size - (e - s), dtype=ct.dtype, device=device)], 1)
            xg = self.meanflow_wrapper.sample(self.meanflow_model, B, ct, self.feat_dim, spk=spkemb, device=device)
            chunks.append(xg.permute(0, 1, 3, 2).squeeze(1)[:, :, :e - s])
        return torch.cat(chunks, dim=2), token_lengths.clone().to(device)

    def training_step(self, batch, batch_idx):
        log_dict = {}
        spkemb = batch['spkemb'].to(self.device)
        tokens = batch['token'].to(self.device)
        tl = batch['token_lengths']
        audio = batch['audio_arrays'].to(self.device)
        al = batch['audio_lengths']

        max_t = self.max_audio_samples // self.samples_per_token
        if tokens.shape[1] > max_t:
            tokens, tl = tokens[:, :max_t], tl.clamp(max=max_t)
        if audio.shape[1] > self.max_audio_samples:
            audio, al = audio[:, :self.max_audio_samples], al.clamp(max=self.max_audio_samples)

        self.meanflow_model.train()
        latents, _ = self.generate_latent_from_meanflow(spkemb, tokens, tl)
        decoded = self.autoencoder.decode(latents, skip_bottleneck=True)
        reals = audio.unsqueeze(1)
        decoded, reals = trim_to_shortest(decoded, reals)
        ml = min(decoded.shape[-1], min(al).item())
        decoded, reals = decoded[:, :, :ml], reals[:, :, :ml]

        if self.global_step >= self.warmup_steps:
            self.warmed_up = True
        opts = self.optimizers()
        opt_gen = opts[0] if isinstance(opts, list) else opts
        opt_disc = opts[1] if isinstance(opts, list) and len(opts) > 1 else None

        loss_adv = feature_matching_distance = torch.tensor(0., device=self.device)
        if self.use_discriminator and self.warmed_up and opt_disc:
            lr, lf = self.discriminator(reals), self.discriminator(decoded.detach())
            dl = sum(get_hinge_losses(lr[0][i], lf[0][i])[0] for i in range(len(lr[0]))) / len(lr[0])
            opt_disc.zero_grad(); self.manual_backward(dl)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            opt_disc.step(); opt_disc.zero_grad(set_to_none=True)
            log_dict['train/disc_loss'] = dl.item()

            lr_g, fr_g = self.discriminator(reals.detach())
            lf_g, ff_g = self.discriminator(decoded)
            ns = len(lr_g)
            for i in range(ns):
                _, adv = get_hinge_losses(lr_g[i], lf_g[i])
                loss_adv = loss_adv + adv / ns
                feature_matching_distance = feature_matching_distance + sum(
                    (a - b).abs().mean() for a, b in zip(fr_g[i], ff_g[i])) / len(fr_g[i]) / ns

        gen_loss, gen_losses = self.losses_gen({
            'reals': reals, 'decoded': decoded,
            'loss_adv': loss_adv, 'feature_matching_distance': feature_matching_distance,
        })
        opt_gen.zero_grad(); self.manual_backward(gen_loss)
        torch.nn.utils.clip_grad_norm_(self.meanflow_model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.autoencoder.decoder.parameters(), 1.0)
        opt_gen.step(); opt_gen.zero_grad(set_to_none=True)

        log_dict['train/gen_loss'] = gen_loss.item()
        for k, v in gen_losses.items():
            log_dict[f'train/{k}'] = v.detach().item()
        self.log_dict(log_dict, prog_bar=True, on_step=True)
        if self.global_step % 50 == 0:
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='End-to-End Joint Refinement')
    parser.add_argument('--meanspeech_parquet_dir', type=str, required=True)
    parser.add_argument('--libritts_parquet_dir', type=str, required=True)
    parser.add_argument('--meanflow_ckpt', type=str, required=True)
    parser.add_argument('--autoencoder_config', type=str, required=True)
    parser.add_argument('--vae_ckpt', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--meanflow_lr', type=float, default=1e-6)
    parser.add_argument('--disc_lr', type=float, default=3e-5)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--feat_dim', type=int, default=16)
    parser.add_argument('--dit_hidden_size', type=int, default=1152)
    parser.add_argument('--dit_depth', type=int, default=28)
    parser.add_argument('--dit_num_heads', type=int, default=16)
    parser.add_argument('--save_dir', type=str, default='./finetune_joint_logs')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    pl.seed_everything(args.seed, workers=True)

    model = JointFinetuneModule(
        meanflow_ckpt_path=args.meanflow_ckpt,
        autoencoder_config_path=args.autoencoder_config,
        vae_ckpt_path=args.vae_ckpt,
        lr=args.lr, meanflow_lr=args.meanflow_lr, disc_lr=args.disc_lr,
        warmup_steps=args.warmup_steps, feat_dim=args.feat_dim,
        dit_hidden_size=args.dit_hidden_size, dit_depth=args.dit_depth, dit_num_heads=args.dit_num_heads,
    )
    dm = pl.LightningDataModule()
    dm.train_dataloader = lambda: get_paired_dataloader(
        args.meanspeech_parquet_dir, args.libritts_parquet_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    logger = TensorBoardLogger(save_dir=args.save_dir, name='joint_refinement')
    trainer = pl.Trainer(
        devices=1, accelerator="gpu", precision='32',
        callbacks=[ModelCheckpoint(every_n_train_steps=20000, dirpath=os.path.join(logger.log_dir, "checkpoints"),
                                   save_top_k=-1, filename='joint_ft-step={step}')],
        logger=logger, log_every_n_steps=1, max_steps=args.max_steps, num_sanity_val_steps=0,
    )
    trainer.fit(model, dm, ckpt_path=args.resume_ckpt)

    torch.save({
        "meanflow_state_dict": model.meanflow_model.state_dict(),
        "decoder_state_dict": model.autoencoder.decoder.state_dict(),
        "autoencoder_state_dict": model.autoencoder.state_dict(),
    }, os.path.join(logger.log_dir, "final_joint_weights.pt"))


if __name__ == '__main__':
    main()
