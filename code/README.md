# Code

Training and fine-tuning code for MeanTok.

## Entry points

| Script | Purpose |
|---|---|
| `speechflow/train_meanflow.py` | Pre-train the MeanFlow latent generator (DiT1D backbone). |
| `finetune_decoder.py` | Decoder-only refinement: freeze MeanFlow, fine-tune the VAE decoder on generated latents. |
| `finetune_joint.py` | End-to-end joint refinement: trainable MeanFlow (low LR) + trainable VAE decoder, using a differentiable one-step sampler so waveform-domain losses propagate through both. |

## Layout

```
code/
├── configs/
│   └── vae_24_25hz.json          # 24-dim / 25Hz VAE architecture config
├── paired_dataloader.py          # (spk_emb, tokens, audio) loader
├── finetune_decoder.py
├── finetune_joint.py
├── speechflow/                   # MeanFlow latent generator
│   ├── config/
│   │   └── 24dim_25hz_small.yaml
│   ├── dataset/                  # arrow-format dataset
│   ├── flow/
│   │   └── meanflow.py           # MeanFlowWrapper + differentiable_sample
│   ├── modules/
│   │   └── dit1d.py              # 1D DiT backbone
│   └── train_meanflow.py
└── stable_audio_tools/           # VAE / audio autoencoder utilities
    ├── configs/
    ├── data/
    ├── inference/
    ├── interface/
    ├── models/
    └── training/
```

## Notes

- All paths in the YAML / JSON configs (`dataset_path`, `model_ckpt_path`, etc.) are
  placeholders. Replace them with your own data and pre-trained VAE checkpoint
  before running.
- `stable_audio_tools/` is adapted from
  [Stability-AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools);
  only the modules used by this project are kept.
- Training uses PyTorch Lightning (`bf16-mixed` precision by default).
