"""Benchmark model size, operation count, latency, and FPS."""

from __future__ import annotations

import argparse
import time

import torch
from thop import profile

from models.model import EfficientUNetWithBiFormerDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = EfficientUNetWithBiFormerDecoder(
        out_channels=2,
        pretrained=False,
    ).to(device)
    model.eval()

    dummy = torch.randn(
        1,
        3,
        args.image_size,
        args.image_size,
        device=device,
    )

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    with torch.no_grad():
        operations, thop_params = profile(
            model,
            inputs=(dummy,),
            verbose=False,
        )

        for _ in range(args.warmup):
            _ = model(dummy)
        synchronize(device)

        elapsed = []
        for _ in range(args.runs):
            synchronize(device)
            start = time.perf_counter()
            _ = model(dummy)
            synchronize(device)
            elapsed.append(time.perf_counter() - start)

    mean_seconds = sum(elapsed) / len(elapsed)
    latency_ms = mean_seconds * 1000.0
    fps = 1.0 / mean_seconds

    print(f"Parameters (model): {total_params / 1e6:.3f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.3f} M")
    print(f"THOP parameters: {thop_params / 1e6:.3f} M")
    print(f"THOP operations: {operations / 1e9:.3f} G")
    print(f"Latency: {latency_ms:.2f} ms/image")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
