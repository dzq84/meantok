"""Paired dataloader: loads (speaker_embedding, tokens, audio) triplets from parquet datasets."""

import os
import torch
import numpy as np
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm


def convert_id_format(libritts_id: str, split: str = 'train') -> str:
    """Convert LibriTTS ID to feature dataset ID format."""
    if split == 'test':
        return libritts_id.replace('_', '-')
    return libritts_id


class PairedDataset(Dataset):
    """Pairs feature data (speaker embeddings, tokens) with LibriTTS audio by ID matching."""

    def __init__(
        self,
        meanspeech_parquet_dir: str,
        libritts_parquet_dir: str,
        meanspeech_split: str = 'train',
        libritts_splits: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        pair_cache_dir: Optional[str] = None,
        use_pair_cache: bool = True,
        max_tokens: int = 50,
        sample_rate: int = 24000,
        token_rate: int = 25,
        random_crop: bool = True,
    ):
        self.meanspeech_split = meanspeech_split
        self.cache_dir = cache_dir
        self.use_pair_cache = use_pair_cache
        self.random_crop = random_crop
        self.max_tokens = max_tokens
        self.sample_rate = sample_rate
        self.token_rate = token_rate
        self.samples_per_token = sample_rate // token_rate

        if libritts_splits is None:
            libritts_splits = (['train.clean.100', 'train.clean.360', 'train.other.500']
                               if meanspeech_split == 'train' else
                               ['test.clean', 'test.other', 'dev.clean', 'dev.other'])
        self.libritts_splits = libritts_splits

        if pair_cache_dir is None:
            pair_cache_dir = os.path.join(meanspeech_parquet_dir, '.pair_cache')
        self.pair_cache_dir = pair_cache_dir
        os.makedirs(self.pair_cache_dir, exist_ok=True)

        cache_key = f"v2_{meanspeech_split}_{'_'.join(sorted(libritts_splits))}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        self.pair_cache_file = os.path.join(self.pair_cache_dir, f"pair_cache_{meanspeech_split}_{cache_hash}.pkl")

        # Load feature parquet
        ms_path = Path(meanspeech_parquet_dir) / meanspeech_split
        ms_files = list(ms_path.glob("*.parquet"))
        self.meanspeech_dataset = load_dataset('parquet', data_files=[str(f) for f in ms_files],
                                                split='train', cache_dir=cache_dir)

        # Load LibriTTS parquet
        lt_files = []
        for sp in libritts_splits:
            sp_path = Path(libritts_parquet_dir) / 'data' / sp
            if sp_path.exists():
                lt_files.extend(list(sp_path.glob("*.parquet")))
        self.libritts_dataset = load_dataset('parquet', data_files=[str(f) for f in lt_files],
                                              split='train', cache_dir=cache_dir)

        if self.use_pair_cache and os.path.exists(self.pair_cache_file):
            with open(self.pair_cache_file, 'rb') as f:
                cache = pickle.load(f)
            self.paired_indices = cache['paired_indices']
            self.libritts_id_to_idx = cache['libritts_id_to_idx']
        else:
            self._build_pair_index()
            if self.use_pair_cache:
                with open(self.pair_cache_file, 'wb') as f:
                    pickle.dump({'paired_indices': self.paired_indices,
                                 'libritts_id_to_idx': self.libritts_id_to_idx,
                                 'meanspeech_split': meanspeech_split,
                                 'libritts_splits': libritts_splits}, f)

    def _build_pair_index(self):
        self.libritts_id_to_idx = {}
        for idx in tqdm(range(len(self.libritts_dataset)), desc="Indexing"):
            self.libritts_id_to_idx[convert_id_format(self.libritts_dataset[idx]['id'], self.meanspeech_split)] = idx
        self.paired_indices = [i for i in range(len(self.meanspeech_dataset))
                               if self.meanspeech_dataset[i]['wav_id'] in self.libritts_id_to_idx]

    def __len__(self):
        return len(self.paired_indices)

    def __getitem__(self, idx):
        ms_item = self.meanspeech_dataset[self.paired_indices[idx]]
        wav_id = ms_item['wav_id']
        spkemb = torch.tensor(ms_item['spkemb'], dtype=torch.float32).reshape(ms_item['spkemb_shape'])
        token = torch.tensor(ms_item['token'], dtype=torch.long).reshape(ms_item['token_shape'])
        if spkemb.dim() == 2 and spkemb.shape[0] == 1:
            spkemb = spkemb.squeeze(0)
        if token.dim() == 2 and token.shape[0] == 1:
            token = token.squeeze(0)

        audio_array = torch.tensor(self.libritts_dataset[self.libritts_id_to_idx[wav_id]]['audio']['array'],
                                   dtype=torch.float32)

        if self.random_crop and token.shape[0] > self.max_tokens:
            start = torch.randint(0, token.shape[0] - self.max_tokens + 1, (1,)).item()
            token = token[start:start + self.max_tokens]
            s_audio = start * self.samples_per_token
            audio_array = audio_array[s_audio:min(s_audio + self.max_tokens * self.samples_per_token, len(audio_array))]
        elif self.random_crop:
            audio_array = audio_array[:min(token.shape[0] * self.samples_per_token, len(audio_array))]

        return {'wav_id': wav_id, 'spkemb': spkemb, 'token': token, 'audio_array': audio_array}


def collate_fn(batch):
    wav_ids = [b['wav_id'] for b in batch]
    spkembs = torch.stack([b['spkemb'] for b in batch])
    tokens = [b['token'] for b in batch]
    tl = torch.tensor([t.shape[0] for t in tokens], dtype=torch.long)
    pt = torch.zeros(len(batch), max(tl), dtype=torch.long)
    for i, t in enumerate(tokens):
        pt[i, :t.shape[0]] = t
    audios = [b['audio_array'] for b in batch]
    al = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)
    pa = torch.zeros(len(batch), max(al), dtype=torch.float32)
    for i, a in enumerate(audios):
        pa[i, :a.shape[0]] = a
    return {'wav_id': wav_ids, 'spkemb': spkembs, 'token': pt, 'token_lengths': tl,
            'audio_arrays': pa, 'audio_lengths': al}


def get_paired_dataloader(meanspeech_parquet_dir, libritts_parquet_dir, meanspeech_split='train',
                          libritts_splits=None, batch_size=32, shuffle=True, num_workers=4,
                          cache_dir=None, pair_cache_dir=None, use_pair_cache=True,
                          pin_memory=True, random_crop=True, max_tokens=50):
    ds = PairedDataset(meanspeech_parquet_dir, libritts_parquet_dir, meanspeech_split, libritts_splits,
                       cache_dir, pair_cache_dir, use_pair_cache, max_tokens=max_tokens, random_crop=random_crop)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_fn, pin_memory=pin_memory)
