# train_resnet18.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn.functional as F
import librosa
import numpy as np
import pandas as pd
import dataset_utils

# 1) Dataset class
class MelDataset(Dataset):
    def __init__(self, df: pd.DataFrame,
                 sr: int = 4000,
                 n_fft: int = 512,
                 hop_length: int = 128,
                 n_mels: int = 64,
                 max_frames: int = 128):
        # include normal, murmur, artifact
        df = df[df.label.isin(['normal', 'murmur', 'artifact'])].reset_index(drop=True)
        self.df = df
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_frames = max_frames
        # map labels to 0,1,2
        self.label_map = {'normal': 0, 'murmur': 1, 'artifact': 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        # mel-spectrogram
        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0
        )
        log_S = librosa.power_to_db(S, ref=np.max)
        spec = torch.from_numpy(log_S).unsqueeze(0)
        # pad/truncate
        T = spec.size(-1)
        if T < self.max_frames:
            spec = F.pad(spec, (0, self.max_frames - T))
        else:
            spec = spec[..., :self.max_frames]
        label = self.label_map[row['label']]
        return spec.float(), label

# 2) Training loop
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for specs, labels in loader:
        specs = specs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * specs.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += specs.size(0)
    return total_loss / total_samples, total_correct / total_samples
