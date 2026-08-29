"""EfficientNetV2-S encoder with SE and BiFormer decoder."""

import torch
import torch.nn as nn
import timm
from timm.layers import DropBlock2d

from .biformer_block import BiFormerDecoderBlock


class SqueezeExcitation(nn.Module):
    """Channel attention applied after decoder skip concatenation."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        weights = self.pool(x).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1, 1)
        return x * weights


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderStage(nn.Module):
    """Upsample, fuse skip features, apply SE, convolution, DropBlock and BiFormer."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        drop_path: float = 0.2,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        fused_channels = skip_channels + out_channels
        self.se = SqueezeExcitation(fused_channels)
        self.merge = nn.Sequential(
            nn.Conv2d(fused_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropblock = DropBlock2d(block_size=3, drop_prob=0.3)
        self.biformer = BiFormerDecoderBlock(
            dim=out_channels,
            drop_path=drop_path,
        )

    @staticmethod
    def _crop_and_concat(
        upsampled: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        if upsampled.shape[2:] != skip.shape[2:]:
            diff_y = skip.size(2) - upsampled.size(2)
            diff_x = skip.size(3) - upsampled.size(3)
            skip = skip[
                :,
                :,
                diff_y // 2 : diff_y // 2 + upsampled.size(2),
                diff_x // 2 : diff_x // 2 + upsampled.size(3),
            ]
        return torch.cat([skip, upsampled], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)
        x = self._crop_and_concat(x, skip)
        x = self.se(x)
        x = self.merge(x)
        x = self.dropblock(x)
        return self.biformer(x)


class EfficientUNetWithBiFormerDecoder(nn.Module):
    """Binary segmentation network used in the manuscript experiments."""

    encoder_name = "tf_efficientnetv2_s.in21k"

    def __init__(
        self,
        out_channels: int = 2,
        pretrained: bool = True,
        drop_path: float = 0.2,
    ) -> None:
        super().__init__()

        self.encoder = timm.create_model(
            self.encoder_name,
            pretrained=pretrained,
            features_only=True,
        )
        encoder_channels = list(self.encoder.feature_info.channels())

        if len(encoder_channels) != 5:
            raise RuntimeError(
                f"Expected 5 EfficientNetV2-S feature stages, got "
                f"{len(encoder_channels)}: {encoder_channels}"
            )

        self.decoder1 = DecoderStage(
            encoder_channels[4],
            encoder_channels[3],
            128,
            drop_path,
        )
        self.decoder2 = DecoderStage(
            128,
            encoder_channels[2],
            64,
            drop_path,
        )
        self.decoder3 = DecoderStage(
            64,
            encoder_channels[1],
            48,
            drop_path,
        )
        self.decoder4 = DecoderStage(
            48,
            encoder_channels[0],
            24,
            drop_path,
        )

        self.final_up = nn.ConvTranspose2d(24, 12, kernel_size=2, stride=2)
        self.final_drop1 = DropBlock2d(block_size=3, drop_prob=0.3)
        self.final_conv1 = DoubleConv(12, 12)
        self.final_drop2 = DropBlock2d(block_size=3, drop_prob=0.3)
        self.final_conv2 = DoubleConv(12, 6)
        self.out_conv = nn.Conv2d(6, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4, x5 = self.encoder(x)

        x = self.decoder1(x5, x4)
        x = self.decoder2(x, x3)
        x = self.decoder3(x, x2)
        x = self.decoder4(x, x1)

        x = self.final_up(x)
        x = self.final_drop1(x)
        x = self.final_conv1(x)
        x = self.final_drop2(x)
        x = self.final_conv2(x)

        return self.out_conv(x)
