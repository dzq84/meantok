# MeanTok: One-Step Token-to-Waveform Generation with MeanFlow in Latent Space

A Token2Wav system that performs **one-step waveform generation** via MeanFlow in a highly
compressed latent space, with refinement strategies (decoder-only and end-to-end joint
fine-tuning) to mitigate latent mismatch.

- **Audio demo**: <https://dzq84.github.io/meantok/demo/>
- **Code**: see [`code/`](./code)

## Repository layout

```
meantok/
├── code/                 # training & fine-tuning code
│   ├── speechflow/       # MeanFlow latent generator (DiT1D backbone)
│   ├── stable_audio_tools/  # VAE / audio autoencoder utilities
│   ├── configs/          # VAE config (vae_24_25hz.json)
│   ├── paired_dataloader.py
│   ├── finetune_decoder.py   # decoder-only refinement
│   └── finetune_joint.py     # end-to-end joint refinement
└── demo/                 # static audio demo page (GitHub Pages)
    ├── index.html
    └── audio/            # wav samples & mel-spectrogram thumbnails
```

## Quick start

This repository is released as a **research reference**. Datasets and pre-trained
checkpoints are not included; paths in the config files are placeholders that you
need to point at your own data and VAE checkpoint.

### 1. Train the MeanFlow latent generator

```bash
cd code
python -m speechflow.train_meanflow \
    --config speechflow/config/24dim_25hz_small.yaml
```

Edit `speechflow/config/24dim_25hz_small.yaml`:
- `dataset_path`: path to your arrow-format paired dataset (speaker embedding, tokens, latents)
- `autoencoder.model_ckpt_path`: path to a pre-trained VAE checkpoint
- `model.feat_dim` / `depth` / `hidden_size`: switch between the 140M small model and the 600M big model

### 2. Decoder-only refinement (freeze MeanFlow, fine-tune VAE decoder)

```bash
python finetune_decoder.py --config <your_config>.yaml
```

### 3. End-to-end joint refinement (trainable MeanFlow + decoder)

```bash
python finetune_joint.py --config <your_config>.yaml
```

The joint script uses a differentiable one-step sampler so that waveform-domain
losses (MRSTFT + adversarial + feature matching) update both the latent generator
and the decoder.

## Demo page

The HTML demo under [`demo/`](./demo) is fully self-contained (audio + mel
thumbnails). To publish it via GitHub Pages, enable Pages on this repo with
`main` / `/ (root)` as the source — the demo will then be served at
<https://dzq84.github.io/meantok/demo/>.

## Acknowledgements

The VAE / autoencoder utilities under `code/stable_audio_tools/` are adapted from
[Stability-AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools).
