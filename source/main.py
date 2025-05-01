# main.py

import dataset_utils
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
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
from train_resnet18 import MelDataset
from train_resnet18 import train_epoch
from s1s2_detector import detect_s1_s2
from features_utils import compute_mel_spectrogram, extract_cycle_spectrograms

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = dataset_utils.load_dataset("/Users/yusufbayindir/Desktop/Tez/data")
    df_B = df[df.source == 'B']
    df_A = df[df.source == 'A']

    batch_size = 16
    loader_B = DataLoader(MelDataset(df_B), batch_size=batch_size, shuffle=True, num_workers=4)
    loader_A = DataLoader(MelDataset(df_A), batch_size=batch_size, shuffle=True, num_workers=4)

    # ResNet18 without pretrained weights
    model = models.resnet18(weights=None)
    # adapt to 1-channel
    model.conv1 = nn.Conv2d(
        1,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False
    )
    # modify classifier to 3 classes
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 5
    # Phase1: train on B
    for epoch in range(1, epochs+1):
        loss, acc = train_epoch(model, loader_B, criterion, optimizer, device)
        print(f"[Phase1][Epoch {epoch}/{epochs}] loss={loss:.4f}, acc={acc:.4f}")
    # Phase2: train on A
    for epoch in range(1, epochs+1):
        loss, acc = train_epoch(model, loader_A, criterion, optimizer, device)
        print(f"[Phase2][Epoch {epoch}/{epochs}] loss={loss:.4f}, acc={acc:.4f}")

    torch.save(model.state_dict(), "resnet18_3class.pt")
    print("Model saved as resnet18_3class.pt")
