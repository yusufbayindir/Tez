# inference.py

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import librosa
import numpy as np

LABELS = ['normal', 'murmur', 'artifact']


def load_model(model_path: str, device: torch.device) -> nn.Module:
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # initialize ResNet18 without pretrained weights
    model = models.resnet18(weights=None)
    # adapt first conv layer to 1 input channel
    model.conv1 = nn.Conv2d(
        1,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False
    )
    # modify classifier to output 3 classes
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print("Model loaded and set to eval mode")
    return model


def preprocess(path: str,
               sr: int = 4000,
               n_fft: int = 512,
               hop_length: int = 128,
               n_mels: int = 64,
               max_frames: int = 128
) -> torch.Tensor:
    print(f"Preprocessing file: {path}")
    y, _ = librosa.load(path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    log_S = librosa.power_to_db(S, ref=np.max)
    spec = torch.from_numpy(log_S).unsqueeze(0).unsqueeze(0)
    T = spec.size(-1)
    if T < max_frames:
        spec = F.pad(spec, (0, max_frames - T))
    else:
        spec = spec[..., :max_frames]
    print(f"  Spectrogram shape: {spec.shape}")
    return spec.float()


def predict(model: nn.Module,
            spec: torch.Tensor,
            device: torch.device
) -> dict:
    spec = spec.to(device)
    with torch.no_grad():
        logits = model(spec)
        print(f"  Raw logits: {logits.cpu().numpy()}")
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return {label: float(probs[i]) for i, label in enumerate(LABELS)}


def main():
    parser = argparse.ArgumentParser(
        description="Inference for 3-class (normal/murmur/artifact) heart sounds"
    )
    parser.add_argument(
        '--model', required=True,
        help='Path to trained model .pt'
    )
    parser.add_argument(
        '--input', required=True,
        help='WAV file or directory'
    )
    args = parser.parse_args()

    print(f"Arguments: model={args.model}, input={args.input}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, device)

    # print model summary
    print("\n=== Model Architecture ===")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # gather input files
    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f)
                 for f in sorted(os.listdir(args.input))
                 if f.lower().endswith('.wav')]
    else:
        files = [args.input]
    print(f"Found {len(files)} file(s) to process")

    # inference loop
    for path in files:
        spec = preprocess(path)
        scores = predict(model, spec, device)
        print(f"\nResults for {path}:")
        for label, prob in scores.items():
            print(f"  {label}: {prob:.4f}")

if __name__ == "__main__":
    main()