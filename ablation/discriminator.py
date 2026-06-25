import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    Least Squares GAN (LSGAN) / PatchGAN Discriminator for Magnitude Spectrograms.
    Input: [B, 1, F, T] where F=257 for n_fft=512.
    Output: Score map predicting Real (1.0) vs Fake (0.0).
    Optionally returns intermediate features for Feature Matching Loss.
    """
    def __init__(self, in_channels=1, ndf=16):
        super().__init__()

        # Sequence of 2D Convolutions
        # Using InstanceNorm2d instead of BatchNorm2d is often more stable for GANs with variable batch sizes.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Output layer
        self.out_conv = nn.Conv2d(ndf * 8, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))

    def forward(self, x, return_features=False):
        """
        x: Magnitude Spectrogram [B, F, T] or [B, 1, F, T]
        return_features: if True, also return intermediate feature maps for FM loss.

        Returns:
            out: PatchGAN score map [B, 1, H, W]
            features (optional): list of 4 intermediate feature tensors
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, F, T]

        features = []
        x = self.conv1(x)
        features.append(x)
        x = self.conv2(x)
        features.append(x)
        x = self.conv3(x)
        features.append(x)
        x = self.conv4(x)
        features.append(x)
        out = self.out_conv(x)

        # Return full spatial score map for proper PatchGAN discrimination.
        if return_features:
            return out, features
        return out


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator inspired by HiFi-GAN / CMGAN.
    Applies independent Discriminators at progressively downsampled input resolutions.
    Each scale provides independent adversarial + feature matching loss signals,
    allowing the model to capture both fine-grained and coarse spectral patterns.
    """
    def __init__(self, num_scales=3, in_channels=1, ndf=16):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            Discriminator(in_channels, ndf) for _ in range(num_scales)
        ])
        # AvgPool to downsample between scales (smoother than MaxPool for spectrograms)
        self.downsample = nn.AvgPool2d(
            kernel_size=(2, 2), stride=(2, 2), count_include_pad=False
        )

    def forward(self, x, return_features=False):
        """
        x: Magnitude Spectrogram [B, F, T] or [B, 1, F, T]

        Returns:
            outputs: list of score maps, one per scale
            features (optional): list of feature lists, one per scale
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        outputs = []
        all_features = []

        for i, disc in enumerate(self.discriminators):
            if return_features:
                out, feats = disc(x, return_features=True)
                outputs.append(out)
                all_features.append(feats)
            else:
                outputs.append(disc(x))

            # Downsample for next scale (not after last discriminator)
            if i < self.num_scales - 1:
                x = self.downsample(x)

        if return_features:
            return outputs, all_features
        return outputs


def adversarial_d_loss(pred_reals, pred_fakes):
    """LSGAN discriminator loss averaged across scales.

    Args:
        pred_reals: list of D outputs on real (clean) spectrograms
        pred_fakes: list of D outputs on fake (enhanced) spectrograms
    """
    loss = 0.0
    for pred_real, pred_fake in zip(pred_reals, pred_fakes):
        loss += 0.5 * F.mse_loss(pred_real, torch.ones_like(pred_real))
        loss += 0.5 * F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
    return loss / len(pred_reals)


def adversarial_g_loss(pred_fakes):
    """LSGAN generator loss averaged across scales.

    Args:
        pred_fakes: list of D outputs on fake (enhanced) spectrograms
    """
    loss = 0.0
    for pred_fake in pred_fakes:
        loss += F.mse_loss(pred_fake, torch.ones_like(pred_fake))
    return loss / len(pred_fakes)


def feature_matching_loss(real_features, fake_features):
    """L1 Feature Matching Loss between real and fake intermediate features.

    Compares D's intermediate representations of real vs fake input,
    providing a smoother training signal than adversarial loss alone.
    Real features are detached to prevent gradient flow to D.

    Args:
        real_features: list (per scale) of lists (per layer) of feature tensors
        fake_features: list (per scale) of lists (per layer) of feature tensors
    """
    loss = 0.0
    num_items = 0
    for real_feat_list, fake_feat_list in zip(real_features, fake_features):
        for real_feat, fake_feat in zip(real_feat_list, fake_feat_list):
            loss += F.l1_loss(fake_feat, real_feat.detach())
            num_items += 1
    return loss / max(num_items, 1)
