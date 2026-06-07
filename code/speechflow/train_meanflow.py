"""MeanFlow latent generator training script (PyTorch Lightning)."""

import os
import sys
import argparse
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torchaudio
import json
import logging

from speechflow.dataset.dataset_arrow import CozyArrowDataset, collate_fn
from speechflow.modules.dit1d import DiT1D
from speechflow.flow.meanflow import MeanFlowWrapper
from stable_audio_tools.models import create_model_from_config


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class MeanFlowTrainer(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-4,
        sample_every_n_steps=1000,
        model_config_path=None,
        model_ckpt_path=None,
        spk_dim=192,
        feat_dim=16,
        input_size=126,
        patch_size=1,
        in_channels=1,
        depth=12,
        hidden_size=768,
        num_heads=12,
        num_tokens=6561,
        exp_name="meanflow_train",
        crop_length=126,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.exp_name = exp_name

        self.model = DiT1D(
            input_size=input_size, patch_size=patch_size, in_channels=in_channels,
            feat_dim=feat_dim, depth=depth, hidden_size=hidden_size,
            num_heads=num_heads, num_tokens=num_tokens, spk_dim=spk_dim,
        )
        self.fm = MeanFlowWrapper(
            channels=1, image_size=126, flow_ratio=0.50,
            time_dist=('lognorm', -0.4, 1.0), jvp_api='autograd',
        )

        self.model_config_path = model_config_path
        self.model_ckpt_path = model_ckpt_path
        self.autoencoder = None
        self.sample_rate = None
        if model_config_path and model_ckpt_path:
            self._load_autoencoder()

    def _load_autoencoder(self):
        with open(self.model_config_path, 'r') as f:
            model_config = json.load(f)
        self.sample_rate = model_config["sample_rate"]
        self.autoencoder = create_model_from_config(model_config).to(self.device)
        ckpt = torch.load(self.model_ckpt_path, map_location=self.device)
        state_dict = ckpt.get("state_dict", ckpt)
        new_sd = {(k[12:] if k.startswith("autoencoder.") else k): v for k, v in state_dict.items()}
        self.autoencoder.load_state_dict(new_sd, strict=False)
        self.autoencoder.eval()

    def forward(self, x, t, y, spk):
        return self.model(x=x, t=t, y=y, spk=spk)

    def training_step(self, batch, batch_idx):
        tokens = batch['token'].squeeze(1)
        latents = batch['latent'].permute(0, 1, 3, 2)
        spk_emb = batch['spk_emb']
        loss, mse_val = self.fm.loss(model=self, x=latents, c=tokens, spk=spk_emb)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_mse', mse_val, prog_bar=True, on_step=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step > 0 and self.global_step % self.hparams.sample_every_n_steps == 0:
            subset = Subset(self.trainer.train_dataloader.dataset, range(10))
            sample_loader = DataLoader(subset, batch_size=10, shuffle=False,
                                       collate_fn=collate_fn, num_workers=0, pin_memory=True)
            self._sample_and_log(next(iter(sample_loader)), "train")

    @torch.no_grad()
    def _sample_and_log(self, batch, dataset_type="train"):
        tokens = batch['token'].squeeze(1).to(self.device)
        x_gt = batch['latent'].permute(0, 1, 3, 2).to(self.device)
        spk_emb = batch['spk_emb'].to(self.device)
        x_gen = self.fm.sample(self, x_gt.shape[0], tokens, self.hparams.feat_dim, spk=spk_emb, device=self.device)

        if self.autoencoder is not None:
            audio_gen = self.autoencoder.decode_audio(x_gen.squeeze(1).permute(0, 2, 1), chunked=False)
            audio_gt = self.autoencoder.decode_audio(x_gt.squeeze(1).permute(0, 2, 1), chunked=False)
            demo_dir = f"./exp_logs/{self.exp_name}/demo/{dataset_type}_step_{self.global_step}"
            os.makedirs(demo_dir, exist_ok=True)
            for i in range(audio_gen.shape[0]):
                torchaudio.save(f"{demo_dir}/sample_{i:02d}_gen.wav", audio_gen[i].cpu(), self.sample_rate)
                torchaudio.save(f"{demo_dir}/sample_{i:02d}_gt.wav", audio_gt[i].cpu(), self.sample_rate)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


def main():
    parser = argparse.ArgumentParser(description='MeanFlow Latent Generator Training')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    config = load_config(args.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']
    logging.basicConfig(level=logging.INFO)

    dataset = CozyArrowDataset(
        dataset_path=config['dataset_path'], split=config.get('split', 'train'),
        crop_length=config['model'].get('input_size', 126), feat_dim=config['model']['feat_dim'],
    )
    train_loader = DataLoader(
        dataset, batch_size=config['training']['batch_size'], shuffle=True,
        num_workers=config['training']['num_workers'], collate_fn=collate_fn,
        pin_memory=True, persistent_workers=config['training']['num_workers'] > 0,
        prefetch_factor=2, drop_last=False,
    )

    model = MeanFlowTrainer(
        learning_rate=config['training']['learning_rate'],
        sample_every_n_steps=config['training']['sample_every_n_steps'],
        model_config_path=config['autoencoder'].get('model_config_path'),
        model_ckpt_path=config['autoencoder'].get('model_ckpt_path'),
        **{k: config['model'][k] for k in ['spk_dim', 'feat_dim', 'input_size', 'patch_size',
                                             'in_channels', 'depth', 'hidden_size', 'num_heads', 'num_tokens']},
        exp_name=config['exp_name'],
        crop_length=config['model'].get('input_size', 126),
    )

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'], accelerator="gpu", devices=1,
        precision=config['training']['precision'],
        log_every_n_steps=config['training']['log_every_n_steps'],
        callbacks=[ModelCheckpoint(dirpath=config['checkpoint']['dirpath'],
                                   filename=config['checkpoint']['filename'],
                                   save_top_k=-1,
                                   every_n_epochs=config['training']['save_every_n_epochs'])],
        logger=TensorBoardLogger(save_dir="exp_logs", name=config['exp_name']),
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
    )
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
