# Demo

Static audio demo page. Open [`index.html`](./index.html) directly in a browser,
or publish via GitHub Pages.

Served at: <https://dzq84.github.io/meantok/demo/>

## Layout

```
demo/
├── index.html       # demo page (self-contained, no external assets)
└── audio/
    ├── gt/                # ground-truth references
    ├── baseline_10step/   # 10-step diffusion baseline
    ├── noFT_d24/          # MeanFlow only (no fine-tuning), latent dim = 24
    ├── decoderFT_d24/     # decoder-only refinement, latent dim = 24
    ├── jointFT_d8 / d16 / d24       # joint refinement at different latent dims
    ├── jointFT_d24_big    # joint refinement, 600M backbone
    └── mel/               # mel-spectrogram thumbnails for every sample
```

## Publishing via GitHub Pages

1. Push this repository to GitHub.
2. In **Settings → Pages**, select **Branch: `main`**, **Folder: `/ (root)`**.
3. The demo will be available at `https://<user>.github.io/meantok/demo/`.
