# -*- coding: utf-8 -*-
"""
Fast AdaIN style transfer model.
"""

from collections import OrderedDict

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Full normalized VGG layout used by common AdaIN checkpoints.
# We only use relu4_1 for AdaIN, but the checkpoint often contains
# layers through relu5_4. Defining the full layout prevents unexpected
# key errors during load.
# ---------------------------------------------------------------------
def make_full_vgg():
    return nn.Sequential(
        nn.Conv2d(3, 3, 1),                         # 0
        nn.ReflectionPad2d((1, 1, 1, 1)),            # 1
        nn.Conv2d(3, 64, 3),                        # 2
        nn.ReLU(inplace=True),                      # 3  relu1_1

        nn.ReflectionPad2d((1, 1, 1, 1)),            # 4
        nn.Conv2d(64, 64, 3),                       # 5
        nn.ReLU(inplace=True),                      # 6  relu1_2
        nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 7

        nn.ReflectionPad2d((1, 1, 1, 1)),            # 8
        nn.Conv2d(64, 128, 3),                      # 9
        nn.ReLU(inplace=True),                      # 10 relu2_1

        nn.ReflectionPad2d((1, 1, 1, 1)),            # 11
        nn.Conv2d(128, 128, 3),                     # 12
        nn.ReLU(inplace=True),                      # 13 relu2_2
        nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 14

        nn.ReflectionPad2d((1, 1, 1, 1)),            # 15
        nn.Conv2d(128, 256, 3),                     # 16
        nn.ReLU(inplace=True),                      # 17 relu3_1

        nn.ReflectionPad2d((1, 1, 1, 1)),            # 18
        nn.Conv2d(256, 256, 3),                     # 19
        nn.ReLU(inplace=True),                      # 20 relu3_2

        nn.ReflectionPad2d((1, 1, 1, 1)),            # 21
        nn.Conv2d(256, 256, 3),                     # 22
        nn.ReLU(inplace=True),                      # 23 relu3_3

        nn.ReflectionPad2d((1, 1, 1, 1)),            # 24
        nn.Conv2d(256, 256, 3),                     # 25
        nn.ReLU(inplace=True),                      # 26 relu3_4
        nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 27

        nn.ReflectionPad2d((1, 1, 1, 1)),            # 28
        nn.Conv2d(256, 512, 3),                     # 29
        nn.ReLU(inplace=True),                      # 30 relu4_1

        nn.ReflectionPad2d((1, 1, 1, 1)),            # 31
        nn.Conv2d(512, 512, 3),                     # 32
        nn.ReLU(inplace=True),                      # 33 relu4_2

        nn.ReflectionPad2d((1, 1, 1, 1)),            # 34
        nn.Conv2d(512, 512, 3),                     # 35
        nn.ReLU(inplace=True),                      # 36 relu4_3

        nn.ReflectionPad2d((1, 1, 1, 1)),            # 37
        nn.Conv2d(512, 512, 3),                     # 38
        nn.ReLU(inplace=True),                      # 39 relu4_4
        nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 40

        nn.ReflectionPad2d((1, 1, 1, 1)),            # 41
        nn.Conv2d(512, 512, 3),                     # 42
        nn.ReLU(inplace=True),                      # 43 relu5_1

        nn.ReflectionPad2d((1, 1, 1, 1)),            # 44
        nn.Conv2d(512, 512, 3),                     # 45
        nn.ReLU(inplace=True),                      # 46 relu5_2

        nn.ReflectionPad2d((1, 1, 1, 1)),            # 47
        nn.Conv2d(512, 512, 3),                     # 48
        nn.ReLU(inplace=True),                      # 49 relu5_3

        nn.ReflectionPad2d((1, 1, 1, 1)),            # 50
        nn.Conv2d(512, 512, 3),                     # 51
        nn.ReLU(inplace=True),                      # 52 relu5_4
    )


# ---------------------------------------------------------------------
# AdaIN decoder matching common decoder.pth weights.
# ---------------------------------------------------------------------
def make_decoder():
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, 3),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="nearest"),

        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, 3),
        nn.ReLU(inplace=True),

        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, 3),
        nn.ReLU(inplace=True),

        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, 3),
        nn.ReLU(inplace=True),

        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, 3),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="nearest"),

        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, 3),
        nn.ReLU(inplace=True),

        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, 3),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="nearest"),

        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, 3),
        nn.ReLU(inplace=True),

        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, 3),
    )


def clean_state_dict(state_dict):
    """
    Handles checkpoints saved as:
      - plain state_dict
      - {'state_dict': state_dict}
      - DataParallel keys prefixed with 'module.'
    """
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    """
    Args:
        feat: [N, C, H, W]
    Returns:
        mean, std: [N, C, 1, 1]
    """
    n, c = feat.shape[:2]
    feat_flat = feat.reshape(n, c, -1)
    feat_mean = feat_flat.mean(dim=2).reshape(n, c, 1, 1)
    feat_std = (feat_flat.var(dim=2, unbiased=False) + eps).sqrt().reshape(n, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(
    content_feat: torch.Tensor,
    style_feat: torch.Tensor,
    content_std_blend: float = 0.35,
):
    """
    Robust AdaIN.

    Pure AdaIN uses only style_std. For some very flat/geometric styles,
    style_std can be tiny in many relu4_1 channels, making the target feature
    nearly constant and producing a uniform decoded image.

    content_std_blend preserves part of the content feature contrast:
        0.00 = original AdaIN
        0.35 = robust default for webcam
        0.70+ = more content structure, weaker style
    """
    content_mean, content_std = calc_mean_std(content_feat)
    style_mean, style_std = calc_mean_std(style_feat)

    # Avoid pathological near-zero style variance.
    min_std = content_std.detach() * 0.05
    style_std = torch.maximum(style_std, min_std)

    # Blend style and content variance to keep spatial contrast.
    if content_std_blend > 0:
        style_std = (1.0 - content_std_blend) * style_std + content_std_blend * content_std

    normalized = (content_feat - content_mean) / content_std
    return normalized * style_std + style_mean


class AdaINStyleTransfer(nn.Module):
    def __init__(self, decoder_path: str, vgg_path: str):
        super().__init__()

        full_encoder = make_full_vgg()
        decoder_net = make_decoder()

        vgg_state = clean_state_dict(torch.load(vgg_path, map_location="cpu"))
        vgg_result = full_encoder.load_state_dict(vgg_state, strict=False)

        if vgg_result.missing_keys:
            print(f"# Warning: missing VGG keys: {vgg_result.missing_keys}")
        if vgg_result.unexpected_keys:
            print(f"# Warning: unexpected VGG keys ignored: {vgg_result.unexpected_keys}")

        decoder_state = clean_state_dict(torch.load(decoder_path, map_location="cpu"))
        decoder_result = decoder_net.load_state_dict(decoder_state, strict=False)

        if decoder_result.missing_keys:
            print(f"# Warning: missing decoder keys: {decoder_result.missing_keys}")
        if decoder_result.unexpected_keys:
            print(f"# Warning: unexpected decoder keys ignored: {decoder_result.unexpected_keys}")

        # AdaIN uses relu4_1 features: through index 30 inclusive.
        self.encoder = nn.Sequential(*list(full_encoder.children())[:31])
        self.decoder = decoder_net

        for module in (self.encoder, self.decoder):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def encode_style(self, style_image: torch.Tensor) -> torch.Tensor:
        return self.encoder(style_image)

    def forward_with_cached_style(
        self,
        content_image: torch.Tensor,
        style_feature: torch.Tensor,
        alpha: float = 1.0,
        content_std_blend: float = 0.35,
    ) -> torch.Tensor:
        content_feature = self.encoder(content_image)
        target_feature = adaptive_instance_normalization(
            content_feature,
            style_feature,
            content_std_blend=content_std_blend,
        )
        target_feature = alpha * target_feature + (1.0 - alpha) * content_feature
        return self.decoder(target_feature)

    def forward(
        self,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        style_feature = self.encode_style(style_image)
        return self.forward_with_cached_style(content_image, style_feature, alpha)
