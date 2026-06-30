"""Parameterized DeepVQE ablation variants.

This module intentionally leaves ``deepvqe.py`` untouched. The Baseline
configuration is state-dict compatible with ``deepvqe.DeepVQE`` and can load
baseline weights with ``strict=True``. Non-baseline variants change module
structure and should load baseline weights with ``strict=False`` only.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from deepvqe import CCM, FE, SubpixelConv2d
from ablation.modules.mamba import MambaStack
from stream.modules.convolution import StreamConv2d


BASE_GRU_HIDDEN = 64 * 9


ABLATION_CONFIGS = {
    "Baseline": {
        "prelu_type": None,
        "dw_residual": False,
        "use_eca_f": False,
        "main_block_eca_f": False,
        "gru_hidden": BASE_GRU_HIDDEN,
    },
    "D1a_gru704": {
        "prelu_type": None,
        "dw_residual": False,
        "use_eca_f": False,
        "main_block_eca_f": False,
        "skip_gate": None,
        "dw_subpixel": False,
        "gru_hidden": 704,
    },
    "D1b_gru768": {
        "prelu_type": None,
        "dw_residual": False,
        "use_eca_f": False,
        "main_block_eca_f": False,
        "skip_gate": None,
        "dw_subpixel": False,
        "gru_hidden": 768,
    },
    "D2_temporal_refine": {
        "prelu_type": None,
        "dw_residual": False,
        "use_eca_f": False,
        "main_block_eca_f": False,
        "skip_gate": None,
        "dw_subpixel": False,
        "gru_hidden": BASE_GRU_HIDDEN,
        "temporal_refine": True,
        "temporal_refine_kernel": 3,
    },
    "Mamba_b2_h384": {
        "prelu_type": None,
        "dw_residual": False,
        "use_eca_f": False,
        "main_block_eca_f": False,
        "skip_gate": None,
        "dw_subpixel": False,
        "gru_hidden": BASE_GRU_HIDDEN,
        "sequence_model": "mamba",
        "mamba_blocks": 2,
        "mamba_hidden": 384,
        "mamba_state": 16,
        "mamba_conv": 4,
        "mamba_expand": 2,
    },
    "B1a": {
        "prelu_type": "shared",
        "dw_residual": False,
        "use_eca_f": False,
        "main_block_eca_f": False,
        "gru_hidden": BASE_GRU_HIDDEN,
    },
    "B1b": {
        "prelu_type": "per_channel",
        "dw_residual": False,
        "use_eca_f": False,
        "main_block_eca_f": False,
        "gru_hidden": BASE_GRU_HIDDEN,
    },
    "B2": {
        "prelu_type": None,
        "dw_residual": False,
        "use_eca_f": True,
        "main_block_eca_f": False,
        "skip_gate": None,
        "dw_subpixel": False,
        "gru_hidden": BASE_GRU_HIDDEN,
    },
    "B3a": {
        "prelu_type": None,
        "dw_residual": False,
        "use_eca_f": False,
        "main_block_eca_f": False,
        "skip_gate": "eca_f",
        "dw_subpixel": False,
        "gru_hidden": BASE_GRU_HIDDEN,
    },
    "B3b": {
        "prelu_type": None,
        "dw_residual": False,
        "use_eca_f": False,
        "main_block_eca_f": False,
        "skip_gate": "se_f",
        "dw_subpixel": False,
        "gru_hidden": BASE_GRU_HIDDEN,
    },
    "C1": {
        "prelu_type": None,
        "dw_residual": True,
        "use_eca_f": False,
        "main_block_eca_f": False,
        "skip_gate": None,
        "dw_subpixel": False,
        "gru_hidden": BASE_GRU_HIDDEN,
    },
    "C2a": {
        "prelu_type": None,
        "dw_residual": True,
        "use_eca_f": False,
        "main_block_eca_f": False,
        "skip_gate": None,
        "dw_subpixel": True,
        "gru_hidden": BASE_GRU_HIDDEN,
    },
    "C2b": {
        "prelu_type": None,
        "dw_residual": True,
        "use_eca_f": True,
        "main_block_eca_f": False,
        "skip_gate": None,
        "dw_subpixel": False,
        "gru_hidden": BASE_GRU_HIDDEN,
    },
    # C3/C4 depend on the B1 tie-breaker. Per-channel matches the previous
    # local default and can be overridden to "shared" once B1a wins.
    "C3": {
        "prelu_type": "per_channel",
        "dw_residual": True,
        "use_eca_f": False,
        "main_block_eca_f": False,
        "skip_gate": None,
        "dw_subpixel": False,
        "gru_hidden": BASE_GRU_HIDDEN,
    },
    "C4": {
        "prelu_type": "per_channel",
        "dw_residual": True,
        "use_eca_f": True,
        "main_block_eca_f": False,
        "skip_gate": None,
        "dw_subpixel": False,
        "gru_hidden": BASE_GRU_HIDDEN,
    },
}


LEGACY_CONFIG_ALIASES = {
    "C1b": "C1",
    "C2": "C2b",
}


LEGACY_CONFIGS = {
    "C1a-g2": {
        "prelu_type": None,
        "dw_residual": False,
        "res_groups": 2,
        "use_eca_f": False,
        "main_block_eca_f": False,
        "skip_gate": None,
        "dw_subpixel": False,
        "gru_hidden": BASE_GRU_HIDDEN,
    },
    "C1a-g4": {
        "prelu_type": None,
        "dw_residual": False,
        "res_groups": 4,
        "use_eca_f": False,
        "main_block_eca_f": False,
        "skip_gate": None,
        "dw_subpixel": False,
        "gru_hidden": BASE_GRU_HIDDEN,
    },
    # Legacy B3 was ECA-F in main encoder/decoder blocks. Roadmap B3a/B3b
    # now cover skip gating, but this remains loadable for older runs.
    "B3": {
        "prelu_type": None,
        "dw_residual": False,
        "use_eca_f": True,
        "main_block_eca_f": True,
        "skip_gate": None,
        "dw_subpixel": False,
        "gru_hidden": BASE_GRU_HIDDEN,
    },
}


def _activation_to_prelu_type(activation):
    if activation in (None, "elu"):
        return None
    if activation == "prelu_shared":
        return "shared"
    if activation == "prelu_channel":
        return "per_channel"
    if activation in ("shared", "per_channel"):
        return activation
    raise ValueError(
        f"Unsupported activation {activation!r}; expected 'elu', "
        "'prelu_shared', or 'prelu_channel'."
    )


def _normalize_model_config(config):
    """Accept planned config keys plus legacy local keys."""
    cfg = deepcopy(config)

    if "activation" in cfg:
        activation = cfg.pop("activation")
        legacy_prelu_type = _activation_to_prelu_type(activation)
        if legacy_prelu_type is not None or cfg.get("prelu_type") is None:
            cfg["prelu_type"] = legacy_prelu_type

    if "attn_type" in cfg:
        attn_type = cfg.pop("attn_type")
        if attn_type in (None, "none"):
            cfg["use_eca_f"] = bool(cfg.get("use_eca_f", False))
        elif attn_type == "eca_f":
            cfg["use_eca_f"] = True
        else:
            raise ValueError(
                "Only attn_type='eca_f' is supported in Phase 1 because "
                "ECA-CT has no streaming cache contract."
            )

    if "res_conv_type" in cfg:
        res_conv_type = cfg.pop("res_conv_type")
        if res_conv_type == "standard":
            cfg["dw_residual"] = bool(cfg.get("dw_residual", False))
        elif res_conv_type == "dw_separable":
            cfg["dw_residual"] = True
        elif res_conv_type == "grouped":
            cfg["dw_residual"] = False
        else:
            raise ValueError(
                f"Unsupported res_conv_type {res_conv_type!r}; expected "
                "'standard', 'grouped', or 'dw_separable'."
            )

    cfg.setdefault("prelu_type", None)
    cfg.setdefault("dw_residual", False)
    cfg.setdefault("use_eca_f", False)
    cfg.setdefault("main_block_eca_f", False)
    cfg.setdefault("gru_hidden", BASE_GRU_HIDDEN)
    cfg.setdefault("res_groups", None)
    cfg.setdefault("skip_gate", None)
    cfg.setdefault("dw_subpixel", False)
    cfg.setdefault("temporal_refine", False)
    cfg.setdefault("temporal_refine_kernel", 3)
    cfg.setdefault("sequence_model", "gru")
    cfg.setdefault("mamba_blocks", 2)
    cfg.setdefault("mamba_hidden", None)
    cfg.setdefault("mamba_state", 16)
    cfg.setdefault("mamba_conv", 4)
    cfg.setdefault("mamba_expand", 2)

    if cfg["skip_gate"] in ("none", "identity", False):
        cfg["skip_gate"] = None
    if cfg["skip_gate"] == "se":
        cfg["skip_gate"] = "se_f"
    if cfg["skip_gate"] not in (None, "eca_f", "se_f"):
        raise ValueError("skip_gate must be None, 'eca_f', or 'se_f'")

    if cfg["sequence_model"] in (None, "rnn"):
        cfg["sequence_model"] = "gru"
    if cfg["sequence_model"] not in ("gru", "mamba"):
        raise ValueError("sequence_model must be 'gru' or 'mamba'")
    cfg["mamba_blocks"] = int(cfg["mamba_blocks"])
    cfg["mamba_state"] = int(cfg["mamba_state"])
    cfg["mamba_conv"] = int(cfg["mamba_conv"])
    cfg["mamba_expand"] = int(cfg["mamba_expand"])
    if cfg["mamba_hidden"] is not None:
        cfg["mamba_hidden"] = int(cfg["mamba_hidden"])
    if cfg["sequence_model"] != "gru" and cfg["temporal_refine"]:
        raise ValueError("temporal_refine is currently only supported with sequence_model='gru'")

    allowed = {
        "prelu_type",
        "dw_residual",
        "use_eca_f",
        "main_block_eca_f",
        "gru_hidden",
        "res_groups",
        "skip_gate",
        "dw_subpixel",
        "temporal_refine",
        "temporal_refine_kernel",
        "sequence_model",
        "mamba_blocks",
        "mamba_hidden",
        "mamba_state",
        "mamba_conv",
        "mamba_expand",
    }
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise ValueError(f"Unknown model config keys: {unknown}")
    return cfg


def get_ablation_config(config_id="Baseline", **overrides):
    """Return a copy of a named ablation config with optional overrides."""
    if config_id in LEGACY_CONFIG_ALIASES:
        config_id = LEGACY_CONFIG_ALIASES[config_id]

    if config_id in ABLATION_CONFIGS:
        config = deepcopy(ABLATION_CONFIGS[config_id])
    elif config_id in LEGACY_CONFIGS:
        config = deepcopy(LEGACY_CONFIGS[config_id])
    else:
        valid = ", ".join(list(ABLATION_CONFIGS) + list(LEGACY_CONFIG_ALIASES) + list(LEGACY_CONFIGS))
        raise ValueError(f"Unknown ablation config {config_id!r}. Valid configs: {valid}")

    config.update(overrides)
    return _normalize_model_config(config)


def ActivationFactory(prelu_type=None, channels=None):
    """Build the configured activation for ``(B, C, T, F)`` feature tensors."""
    prelu_type = _activation_to_prelu_type(prelu_type)
    if prelu_type is None:
        return nn.ELU()
    if prelu_type == "shared":
        return nn.PReLU(num_parameters=1)
    if prelu_type == "per_channel":
        if channels is None:
            raise ValueError("channels must be provided for prelu_type='per_channel'")
        return nn.PReLU(num_parameters=int(channels))
    raise ValueError("prelu_type must be None, 'shared', or 'per_channel'")


class CausalECA_F(nn.Module):
    """Frequency-pooled efficient channel attention.

    Pooling is only over the frequency axis. The 1-D convolution slides across
    channels for each frame independently, so no temporal cache is required.
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("CausalECA_F kernel_size must be odd")
        self.channels = int(channels)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """x: (B, C, T, F)."""
        b, c, t, _ = x.shape
        if c != self.channels:
            raise ValueError(f"CausalECA_F expected {self.channels} channels, got {c}")

        y = x.mean(dim=3, keepdim=True)  # (B, C, T, 1)
        y = y.squeeze(3).permute(0, 2, 1).reshape(b * t, 1, c)
        y = self.sigmoid(self.conv(y))
        y = y.reshape(b, t, c).permute(0, 2, 1).unsqueeze(3)
        return x * y


ECA_F = CausalECA_F


def _attention(channels, enabled):
    return CausalECA_F(channels) if enabled else nn.Identity()


class FrequencySE(nn.Module):
    """Frequency-only squeeze/excitation for causal skip gating."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, int(channels) // int(reduction))
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = x.mean(dim=3, keepdim=True)
        return x * self.gate(y)


def _skip_gate(channels, gate_type):
    if gate_type is None:
        return nn.Identity()
    if gate_type == "eca_f":
        return CausalECA_F(channels)
    if gate_type == "se_f":
        return FrequencySE(channels)
    raise ValueError(f"Unsupported skip_gate={gate_type!r}")


class CausalTemporalRefine(nn.Module):
    """Causal temporal residual refinement over flattened bottleneck features."""

    def __init__(self, features, kernel_size=3):
        super().__init__()
        features = int(features)
        kernel_size = int(kernel_size)
        if kernel_size < 1:
            raise ValueError("temporal_refine_kernel must be >= 1")
        self.features = features
        self.kernel_size = kernel_size
        self.norm = nn.LayerNorm(features)
        self.depthwise = nn.Conv1d(
            features,
            features,
            kernel_size=kernel_size,
            groups=features,
            bias=False,
        )
        self.pointwise = nn.Conv1d(features, features, kernel_size=1)
        self.elu = nn.ELU()

    def forward(self, y):
        """y: (B, T, D). Returns residual delta with the same shape."""
        if y.shape[-1] != self.features:
            raise ValueError(f"CausalTemporalRefine expected {self.features} features, got {y.shape[-1]}")
        z = self.norm(y).transpose(1, 2)
        z = F.pad(z, (self.kernel_size - 1, 0))
        z = self.depthwise(z)
        z = self.pointwise(z)
        z = self.elu(z)
        return z.transpose(1, 2)


class DWSubpixelConv2d(nn.Module):
    """Depthwise-separable version of the original SubpixelConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size=(4, 3)):
        super().__init__()
        self.pad = nn.ZeroPad2d([1, 1, 3, 0])
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels * 2, kernel_size=1)

    def forward(self, x):
        y = self.pointwise(self.depthwise(self.pad(x)))
        y = rearrange(y, "b (r c) t f -> b c t (r f)", r=2)
        return y


class ResidualBlock_Ablation(nn.Module):
    def __init__(
        self,
        channels,
        prelu_type=None,
        dw_residual=False,
        use_eca_f=False,
        res_groups=None,
    ):
        super().__init__()
        channels = int(channels)
        self.pad = nn.ZeroPad2d([1, 1, 3, 0])
        self.dw_residual = bool(dw_residual)
        self.res_groups = res_groups

        if self.dw_residual:
            self.depthwise = nn.Conv2d(channels, channels, kernel_size=(4, 3), groups=channels)
            self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        else:
            groups = int(res_groups) if res_groups is not None else 1
            if channels % groups != 0:
                raise ValueError(f"channels={channels} is not divisible by res_groups={groups}")
            self.conv = nn.Conv2d(channels, channels, kernel_size=(4, 3), groups=groups)

        self.bn = nn.BatchNorm2d(channels)
        self.elu = ActivationFactory(prelu_type, channels)
        self.eca_f = _attention(channels, use_eca_f)

    def forward(self, x):
        """x: (B, C, T, F)."""
        if self.dw_residual:
            y = self.pointwise(self.depthwise(self.pad(x)))
        else:
            y = self.conv(self.pad(x))
        y = self.eca_f(self.elu(self.bn(y)))
        return y + x


class EncoderBlock_Ablation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(4, 3),
        stride=(1, 2),
        prelu_type=None,
        dw_residual=False,
        use_eca_f=False,
        main_block_eca_f=False,
        res_groups=None,
    ):
        super().__init__()
        self.pad = nn.ZeroPad2d([1, 1, 3, 0])
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = ActivationFactory(prelu_type, out_channels)
        self.main_eca_f = _attention(out_channels, main_block_eca_f)
        self.resblock = ResidualBlock_Ablation(
            out_channels,
            prelu_type=prelu_type,
            dw_residual=dw_residual,
            use_eca_f=use_eca_f,
            res_groups=res_groups,
        )

    def forward(self, x):
        y = self.elu(self.bn(self.conv(self.pad(x))))
        y = self.main_eca_f(y)
        return self.resblock(y)


class Bottleneck_Ablation(nn.Module):
    def __init__(self, input_size, hidden_size, temporal_refine=False, temporal_refine_kernel=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        self.refine = (
            CausalTemporalRefine(input_size, temporal_refine_kernel)
            if temporal_refine
            else None
        )

    def forward(self, x):
        """x: (B, C, T, F)."""
        y = rearrange(x, "b c t f -> b t (c f)")
        y = self.gru(y)[0]
        y = self.fc(y)
        if self.refine is not None:
            y = y + self.refine(y)
        y = rearrange(y, "b t (c f) -> b c t f", c=x.shape[1])
        return y


class MambaBottleneck_Ablation(nn.Module):
    def __init__(
        self,
        input_size,
        num_blocks=2,
        hidden_size=None,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.stack = MambaStack(
            self.input_size,
            num_blocks=num_blocks,
            hidden_size=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        """x: (B, C, T, F)."""
        y = rearrange(x, "b c t f -> b t (c f)")
        y, _ = self.stack(y)
        y = rearrange(y, "b t (c f) -> b c t f", c=x.shape[1])
        return y


class DecoderBlock_Ablation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(4, 3),
        is_last=False,
        prelu_type=None,
        dw_residual=False,
        use_eca_f=False,
        main_block_eca_f=False,
        res_groups=None,
        skip_gate=None,
        dw_subpixel=False,
    ):
        super().__init__()
        self.skip_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.skip_gate = _skip_gate(in_channels, skip_gate)
        self.resblock = ResidualBlock_Ablation(
            in_channels,
            prelu_type=prelu_type,
            dw_residual=dw_residual,
            use_eca_f=use_eca_f,
            res_groups=res_groups,
        )
        if dw_subpixel:
            self.deconv = DWSubpixelConv2d(in_channels, out_channels, kernel_size)
        else:
            self.deconv = SubpixelConv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = ActivationFactory(prelu_type, out_channels)
        self.is_last = is_last
        self.main_eca_f = _attention(out_channels, main_block_eca_f and not is_last)

    def forward(self, x, x_en):
        y = x + self.skip_gate(self.skip_conv(x_en))
        y = self.deconv(self.resblock(y))
        if not self.is_last:
            y = self.elu(self.bn(y))
            y = self.main_eca_f(y)
        return y


class DeepVQE_Ablation(nn.Module):
    def __init__(
        self,
        prelu_type=None,
        dw_residual=False,
        use_eca_f=False,
        main_block_eca_f=False,
        gru_hidden=BASE_GRU_HIDDEN,
        skip_gate=None,
        dw_subpixel=False,
        temporal_refine=False,
        temporal_refine_kernel=3,
        sequence_model="gru",
        mamba_blocks=2,
        mamba_hidden=None,
        mamba_state=16,
        mamba_conv=4,
        mamba_expand=2,
        **legacy_kwargs,
    ):
        super().__init__()
        cfg = _normalize_model_config(
            {
                "prelu_type": prelu_type,
                "dw_residual": dw_residual,
                "use_eca_f": use_eca_f,
                "main_block_eca_f": main_block_eca_f,
                "gru_hidden": gru_hidden,
                "skip_gate": skip_gate,
                "dw_subpixel": dw_subpixel,
                "temporal_refine": temporal_refine,
                "temporal_refine_kernel": temporal_refine_kernel,
                "sequence_model": sequence_model,
                "mamba_blocks": mamba_blocks,
                "mamba_hidden": mamba_hidden,
                "mamba_state": mamba_state,
                "mamba_conv": mamba_conv,
                "mamba_expand": mamba_expand,
                **legacy_kwargs,
            }
        )
        self.ablation_config = deepcopy(cfg)

        block_kwargs = {
            "prelu_type": cfg["prelu_type"],
            "dw_residual": cfg["dw_residual"],
            "use_eca_f": cfg["use_eca_f"],
            "main_block_eca_f": cfg["main_block_eca_f"],
            "res_groups": cfg["res_groups"],
        }
        decoder_kwargs = {
            **block_kwargs,
            "skip_gate": cfg["skip_gate"],
            "dw_subpixel": cfg["dw_subpixel"],
        }

        self.fe = FE()
        self.enblock1 = EncoderBlock_Ablation(2, 64, **block_kwargs)
        self.enblock2 = EncoderBlock_Ablation(64, 128, **block_kwargs)
        self.enblock3 = EncoderBlock_Ablation(128, 128, **block_kwargs)
        self.enblock4 = EncoderBlock_Ablation(128, 128, **block_kwargs)
        self.enblock5 = EncoderBlock_Ablation(128, 128, **block_kwargs)

        if cfg["sequence_model"] == "mamba":
            self.bottle = MambaBottleneck_Ablation(
                128 * 9,
                num_blocks=cfg["mamba_blocks"],
                hidden_size=cfg["mamba_hidden"],
                d_state=cfg["mamba_state"],
                d_conv=cfg["mamba_conv"],
                expand=cfg["mamba_expand"],
            )
        else:
            self.bottle = Bottleneck_Ablation(
                128 * 9,
                int(cfg["gru_hidden"]),
                temporal_refine=cfg["temporal_refine"],
                temporal_refine_kernel=cfg["temporal_refine_kernel"],
            )

        self.deblock5 = DecoderBlock_Ablation(128, 128, **decoder_kwargs)
        self.deblock4 = DecoderBlock_Ablation(128, 128, **decoder_kwargs)
        self.deblock3 = DecoderBlock_Ablation(128, 128, **decoder_kwargs)
        self.deblock2 = DecoderBlock_Ablation(128, 64, **decoder_kwargs)
        # Keep the original final activation for Baseline parity, but never
        # attach main-block ECA-F to the output mask branch.
        last_kwargs = dict(decoder_kwargs)
        last_kwargs["main_block_eca_f"] = False
        self.deblock1 = DecoderBlock_Ablation(64, 27, **last_kwargs)
        self.ccm = CCM()

    @classmethod
    def from_config_id(cls, config_id, **overrides):
        return cls(**get_ablation_config(config_id, **overrides))

    def forward(self, x):
        """x: (B, F, T, 2)."""
        en_x0 = self.fe(x)
        en_x1 = self.enblock1(en_x0)
        en_x2 = self.enblock2(en_x1)
        en_x3 = self.enblock3(en_x2)
        en_x4 = self.enblock4(en_x3)
        en_x5 = self.enblock5(en_x4)

        en_xr = self.bottle(en_x5)

        de_x5 = self.deblock5(en_xr, en_x5)[..., : en_x4.shape[-1]]
        de_x4 = self.deblock4(de_x5, en_x4)[..., : en_x3.shape[-1]]
        de_x3 = self.deblock3(de_x4, en_x3)[..., : en_x2.shape[-1]]
        de_x2 = self.deblock2(de_x3, en_x2)[..., : en_x1.shape[-1]]
        de_x1 = self.deblock1(de_x2, en_x1)[..., : en_x0.shape[-1]]

        return self.ccm(de_x1, x)


class StreamResidualBlock_Ablation(nn.Module):
    def __init__(
        self,
        channels,
        prelu_type=None,
        dw_residual=False,
        use_eca_f=False,
        res_groups=None,
    ):
        super().__init__()
        channels = int(channels)
        self.dw_residual = bool(dw_residual)
        self.res_groups = res_groups

        if self.dw_residual:
            self.depthwise = StreamConv2d(
                channels,
                channels,
                kernel_size=(4, 3),
                padding=(0, 1),
                groups=channels,
            )
            self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        else:
            groups = int(res_groups) if res_groups is not None else 1
            if channels % groups != 0:
                raise ValueError(f"channels={channels} is not divisible by res_groups={groups}")
            self.conv = StreamConv2d(
                channels,
                channels,
                kernel_size=(4, 3),
                padding=(0, 1),
                groups=groups,
            )

        self.bn = nn.BatchNorm2d(channels)
        self.elu = ActivationFactory(prelu_type, channels)
        self.eca_f = _attention(channels, use_eca_f)

    def forward(self, x, cache):
        """x: (B, C, 1, F), cache: (B, C, 3, F)."""
        if self.dw_residual:
            y, cache = self.depthwise(x, cache)
            y = self.pointwise(y)
        else:
            y, cache = self.conv(x, cache)
        y = self.eca_f(self.elu(self.bn(y)))
        return y + x, cache


class StreamEncoderBlock_Ablation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(4, 3),
        stride=(1, 2),
        prelu_type=None,
        dw_residual=False,
        use_eca_f=False,
        main_block_eca_f=False,
        res_groups=None,
    ):
        super().__init__()
        self.conv = StreamConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(0, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = ActivationFactory(prelu_type, out_channels)
        self.main_eca_f = _attention(out_channels, main_block_eca_f)
        self.resblock = StreamResidualBlock_Ablation(
            out_channels,
            prelu_type=prelu_type,
            dw_residual=dw_residual,
            use_eca_f=use_eca_f,
            res_groups=res_groups,
        )

    def forward(self, x, conv_cache, res_cache):
        x, conv_cache = self.conv(x, conv_cache)
        x = self.main_eca_f(self.elu(self.bn(x)))
        x, res_cache = self.resblock(x, res_cache)
        return x, conv_cache, res_cache


class StreamBottleneck_Ablation(nn.Module):
    def __init__(self, input_size, hidden_size, temporal_refine=False, temporal_refine_kernel=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        self.refine = (
            CausalTemporalRefine(input_size, temporal_refine_kernel)
            if temporal_refine
            else None
        )
        self.refine_cache_size = int(temporal_refine_kernel) - 1 if temporal_refine else 0

    def forward(self, x, cache, refine_cache=None):
        """x: (B, C, 1, F), cache: (1, B, hidden_size)."""
        y = rearrange(x, "b c t f -> b t (c f)")
        y, cache = self.gru(y, cache)
        y = self.fc(y)
        if self.refine is not None:
            if refine_cache is None:
                raise ValueError("refine_cache is required when temporal_refine=True")
            if refine_cache.shape[2] != self.refine_cache_size:
                raise ValueError(
                    f"Expected refine_cache length {self.refine_cache_size}, got {refine_cache.shape[2]}"
                )
            z = self.refine.norm(y).transpose(1, 2)
            if self.refine_cache_size > 0:
                z_with_history = torch.cat([refine_cache, z], dim=2)
                refine_cache = z_with_history[:, :, -self.refine_cache_size :].contiguous()
            else:
                z_with_history = z
            z = self.refine.depthwise(z_with_history)
            z = self.refine.pointwise(z)
            z = self.refine.elu(z)
            y = y + z.transpose(1, 2)
        y = rearrange(y, "b t (c f) -> b c t f", f=x.shape[-1])
        return y, cache, refine_cache


class StreamMambaBottleneck_Ablation(nn.Module):
    def __init__(
        self,
        input_size,
        num_blocks=2,
        hidden_size=None,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.refine = None
        self.stack = MambaStack(
            self.input_size,
            num_blocks=num_blocks,
            hidden_size=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def cache_names(self):
        return self.stack.cache_names("mamba")

    def init_cache(self, batch_size, device=None, dtype=None):
        return list(self.stack.init_cache(batch_size, device=device, dtype=dtype))

    def forward(self, x, cache, refine_cache=None):
        """x: (B, C, 1, F), cache: flat Mamba cache list."""
        y = rearrange(x, "b c t f -> b t (c f)")
        if y.shape[1] != 1:
            raise ValueError("StreamMambaBottleneck_Ablation expects one frame at a time")
        y, cache = self.stack.step(y[:, 0], tuple(cache))
        y = rearrange(y.unsqueeze(1), "b t (c f) -> b c t f", f=x.shape[-1])
        return y, list(cache), refine_cache


class StreamSubpixelConv2d_Ablation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4, 3)):
        super().__init__()
        self.conv = StreamConv2d(in_channels, out_channels * 2, kernel_size, padding=(0, 1))

    def forward(self, x, cache):
        """x: (B, C, 1, F), cache: (B, C, 3, F)."""
        y, cache = self.conv(x, cache)
        y = rearrange(y, "b (r c) t f -> b c t (r f)", r=2)
        return y, cache


class StreamDWSubpixelConv2d_Ablation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4, 3)):
        super().__init__()
        self.depthwise = StreamConv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=(0, 1),
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels * 2, kernel_size=1)

    def forward(self, x, cache):
        """x: (B, C, 1, F), cache: (B, C, 3, F)."""
        y, cache = self.depthwise(x, cache)
        y = self.pointwise(y)
        y = rearrange(y, "b (r c) t f -> b c t (r f)", r=2)
        return y, cache


class StreamDecoderBlock_Ablation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(4, 3),
        is_last=False,
        prelu_type=None,
        dw_residual=False,
        use_eca_f=False,
        main_block_eca_f=False,
        res_groups=None,
        skip_gate=None,
        dw_subpixel=False,
    ):
        super().__init__()
        self.skip_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.skip_gate = _skip_gate(in_channels, skip_gate)
        self.resblock = StreamResidualBlock_Ablation(
            in_channels,
            prelu_type=prelu_type,
            dw_residual=dw_residual,
            use_eca_f=use_eca_f,
            res_groups=res_groups,
        )
        if dw_subpixel:
            self.deconv = StreamDWSubpixelConv2d_Ablation(in_channels, out_channels, kernel_size)
        else:
            self.deconv = StreamSubpixelConv2d_Ablation(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = ActivationFactory(prelu_type, out_channels)
        self.is_last = is_last
        self.main_eca_f = _attention(out_channels, main_block_eca_f and not is_last)

    def forward(self, x, x_en, conv_cache, res_cache):
        y = x + self.skip_gate(self.skip_conv(x_en))
        y, res_cache = self.resblock(y, res_cache)
        y, conv_cache = self.deconv(y, conv_cache)
        if not self.is_last:
            y = self.elu(self.bn(y))
            y = self.main_eca_f(y)
        return y, conv_cache, res_cache


class StreamCCM_Ablation(nn.Module):
    """Stateful Complex Convolving Mask block."""

    def __init__(self):
        super().__init__()
        self.v = torch.tensor(
            [[1.0, -0.5, -0.5], [0.0, 0.8660254037844386, -0.8660254037844386]],
            dtype=torch.float32,
        )
        self.unfold = nn.Unfold(kernel_size=(3, 3), padding=(0, 1))

    def forward(self, m, x, cache):
        """
        m: (B, 27, 1, F)
        x: (B, F, 1, 2)
        cache: (B, F, 2, 2)
        """
        m = rearrange(m, "b (r c) t f -> b r c t f", r=3)
        v = self.v.to(device=m.device, dtype=m.dtype)
        h_real = torch.sum(v[0][None, :, None, None, None] * m, dim=1)
        h_imag = torch.sum(v[1][None, :, None, None, None] * m, dim=1)

        m_real = rearrange(h_real, "b (m n) t f -> b m n t f", m=3)
        m_imag = rearrange(h_imag, "b (m n) t f -> b m n t f", m=3)

        x = torch.cat([cache, x], dim=2)
        cache = x[:, :, 1:].contiguous()
        x = x.permute(0, 3, 2, 1).contiguous()

        x_unfold = self.unfold(x)
        x_unfold = rearrange(
            x_unfold,
            "b (c m n) (t f) -> b c m n t f",
            c=2,
            m=3,
            n=3,
            f=x.shape[-1],
        )

        x_enh_real = torch.sum(m_real * x_unfold[:, 0] - m_imag * x_unfold[:, 1], dim=(1, 2))
        x_enh_imag = torch.sum(m_real * x_unfold[:, 1] + m_imag * x_unfold[:, 0], dim=(1, 2))
        x_enh = torch.stack([x_enh_real, x_enh_imag], dim=3).transpose(1, 2).contiguous()
        return x_enh, cache


class StreamDeepVQE_Ablation(nn.Module):
    """Stateful streaming counterpart for every ``DeepVQE_Ablation`` variant."""

    cache_prefix_names = (
        "en_conv_cache1",
        "en_res_cache1",
        "en_conv_cache2",
        "en_res_cache2",
        "en_conv_cache3",
        "en_res_cache3",
        "en_conv_cache4",
        "en_res_cache4",
        "en_conv_cache5",
        "en_res_cache5",
    )
    cache_suffix_names = (
        "de_conv_cache5",
        "de_res_cache5",
        "de_conv_cache4",
        "de_res_cache4",
        "de_conv_cache3",
        "de_res_cache3",
        "de_conv_cache2",
        "de_res_cache2",
        "de_conv_cache1",
        "de_res_cache1",
        "m_cache",
    )

    def get_cache_names(self):
        bottle_names = (
            self.bottle.cache_names()
            if hasattr(self.bottle, "cache_names")
            else ("h_cache",)
        )
        names = self.cache_prefix_names + tuple(bottle_names)
        if getattr(self.bottle, "refine", None) is not None:
            names = names + ("temporal_refine_cache",)
        return names + self.cache_suffix_names

    def __init__(
        self,
        prelu_type=None,
        dw_residual=False,
        use_eca_f=False,
        main_block_eca_f=False,
        gru_hidden=BASE_GRU_HIDDEN,
        skip_gate=None,
        dw_subpixel=False,
        temporal_refine=False,
        temporal_refine_kernel=3,
        sequence_model="gru",
        mamba_blocks=2,
        mamba_hidden=None,
        mamba_state=16,
        mamba_conv=4,
        mamba_expand=2,
        **legacy_kwargs,
    ):
        super().__init__()
        cfg = _normalize_model_config(
            {
                "prelu_type": prelu_type,
                "dw_residual": dw_residual,
                "use_eca_f": use_eca_f,
                "main_block_eca_f": main_block_eca_f,
                "gru_hidden": gru_hidden,
                "skip_gate": skip_gate,
                "dw_subpixel": dw_subpixel,
                "temporal_refine": temporal_refine,
                "temporal_refine_kernel": temporal_refine_kernel,
                "sequence_model": sequence_model,
                "mamba_blocks": mamba_blocks,
                "mamba_hidden": mamba_hidden,
                "mamba_state": mamba_state,
                "mamba_conv": mamba_conv,
                "mamba_expand": mamba_expand,
                **legacy_kwargs,
            }
        )
        self.ablation_config = deepcopy(cfg)

        block_kwargs = {
            "prelu_type": cfg["prelu_type"],
            "dw_residual": cfg["dw_residual"],
            "use_eca_f": cfg["use_eca_f"],
            "main_block_eca_f": cfg["main_block_eca_f"],
            "res_groups": cfg["res_groups"],
        }
        decoder_kwargs = {
            **block_kwargs,
            "skip_gate": cfg["skip_gate"],
            "dw_subpixel": cfg["dw_subpixel"],
        }

        self.fe = FE()
        self.enblock1 = StreamEncoderBlock_Ablation(2, 64, **block_kwargs)
        self.enblock2 = StreamEncoderBlock_Ablation(64, 128, **block_kwargs)
        self.enblock3 = StreamEncoderBlock_Ablation(128, 128, **block_kwargs)
        self.enblock4 = StreamEncoderBlock_Ablation(128, 128, **block_kwargs)
        self.enblock5 = StreamEncoderBlock_Ablation(128, 128, **block_kwargs)

        if cfg["sequence_model"] == "mamba":
            self.bottle = StreamMambaBottleneck_Ablation(
                128 * 9,
                num_blocks=cfg["mamba_blocks"],
                hidden_size=cfg["mamba_hidden"],
                d_state=cfg["mamba_state"],
                d_conv=cfg["mamba_conv"],
                expand=cfg["mamba_expand"],
            )
        else:
            self.bottle = StreamBottleneck_Ablation(
                128 * 9,
                int(cfg["gru_hidden"]),
                temporal_refine=cfg["temporal_refine"],
                temporal_refine_kernel=cfg["temporal_refine_kernel"],
            )

        self.deblock5 = StreamDecoderBlock_Ablation(128, 128, **decoder_kwargs)
        self.deblock4 = StreamDecoderBlock_Ablation(128, 128, **decoder_kwargs)
        self.deblock3 = StreamDecoderBlock_Ablation(128, 128, **decoder_kwargs)
        self.deblock2 = StreamDecoderBlock_Ablation(128, 64, **decoder_kwargs)
        last_kwargs = dict(decoder_kwargs)
        last_kwargs["main_block_eca_f"] = False
        self.deblock1 = StreamDecoderBlock_Ablation(64, 27, **last_kwargs)
        self.ccm = StreamCCM_Ablation()

    @classmethod
    def from_config_id(cls, config_id, **overrides):
        return cls(**get_ablation_config(config_id, **overrides))

    @classmethod
    def from_offline(cls, model, strict=True):
        stream_model = cls(**model.ablation_config)
        convert_ablation_to_stream(stream_model, model, strict=strict)
        stream_model.train(model.training)
        return stream_model

    def init_cache(self, batch_size=1, freq_bins=257, device=None, dtype=None):
        """Create zero caches in the fixed order expected by ``forward``."""
        param = next(self.parameters())
        device = device if device is not None else param.device
        dtype = dtype if dtype is not None else param.dtype
        b = int(batch_size)
        f0 = int(freq_bins)
        f1 = (f0 + 1) // 2
        f2 = (f1 + 1) // 2
        f3 = (f2 + 1) // 2
        f4 = (f3 + 1) // 2
        f5 = (f4 + 1) // 2
        if hasattr(self.bottle, "init_cache"):
            bottle_cache = self.bottle.init_cache(b, device=device, dtype=dtype)
        else:
            hidden = self.bottle.gru.hidden_size
            bottle_cache = [torch.zeros(1, b, hidden, device=device, dtype=dtype)]

        cache = [
            torch.zeros(b, 2, 3, f0, device=device, dtype=dtype),
            torch.zeros(b, 64, 3, f1, device=device, dtype=dtype),
            torch.zeros(b, 64, 3, f1, device=device, dtype=dtype),
            torch.zeros(b, 128, 3, f2, device=device, dtype=dtype),
            torch.zeros(b, 128, 3, f2, device=device, dtype=dtype),
            torch.zeros(b, 128, 3, f3, device=device, dtype=dtype),
            torch.zeros(b, 128, 3, f3, device=device, dtype=dtype),
            torch.zeros(b, 128, 3, f4, device=device, dtype=dtype),
            torch.zeros(b, 128, 3, f4, device=device, dtype=dtype),
            torch.zeros(b, 128, 3, f5, device=device, dtype=dtype),
            *bottle_cache,
            torch.zeros(b, 128, 3, f5, device=device, dtype=dtype),
            torch.zeros(b, 128, 3, f5, device=device, dtype=dtype),
            torch.zeros(b, 128, 3, f4, device=device, dtype=dtype),
            torch.zeros(b, 128, 3, f4, device=device, dtype=dtype),
            torch.zeros(b, 128, 3, f3, device=device, dtype=dtype),
            torch.zeros(b, 128, 3, f3, device=device, dtype=dtype),
            torch.zeros(b, 128, 3, f2, device=device, dtype=dtype),
            torch.zeros(b, 128, 3, f2, device=device, dtype=dtype),
            torch.zeros(b, 64, 3, f1, device=device, dtype=dtype),
            torch.zeros(b, 64, 3, f1, device=device, dtype=dtype),
            torch.zeros(b, f0, 2, 2, device=device, dtype=dtype),
        ]
        if self.bottle.refine is not None:
            refine_cache = torch.zeros(
                b,
                self.bottle.refine.features,
                self.bottle.refine_cache_size,
                device=device,
                dtype=dtype,
            )
            cache.insert(len(self.cache_prefix_names) + len(bottle_cache), refine_cache)
        return cache

    def forward(self, x, cache):
        """
        x: (B, F, 1, 2)
        cache: list of tensors following ``cache_names``.
        """
        cache_names = self.get_cache_names()
        if len(cache) != len(cache_names):
            raise ValueError(f"Expected {len(cache_names)} cache tensors, got {len(cache)}")

        prefix_len = len(self.cache_prefix_names)
        bottle_len = len(self.bottle.cache_names()) if hasattr(self.bottle, "cache_names") else 1
        (
            en_conv_cache1,
            en_res_cache1,
            en_conv_cache2,
            en_res_cache2,
            en_conv_cache3,
            en_res_cache3,
            en_conv_cache4,
            en_res_cache4,
            en_conv_cache5,
            en_res_cache5,
        ) = cache[:prefix_len]
        bottle_cache = list(cache[prefix_len : prefix_len + bottle_len])
        idx = prefix_len + bottle_len
        if self.bottle.refine is not None:
            temporal_refine_cache = cache[idx]
            idx += 1
        else:
            temporal_refine_cache = None
        (
            de_conv_cache5,
            de_res_cache5,
            de_conv_cache4,
            de_res_cache4,
            de_conv_cache3,
            de_res_cache3,
            de_conv_cache2,
            de_res_cache2,
            de_conv_cache1,
            de_res_cache1,
            m_cache,
        ) = cache[idx:]

        en_x0 = self.fe(x)
        en_x1, en_conv_cache1, en_res_cache1 = self.enblock1(en_x0, en_conv_cache1, en_res_cache1)
        en_x2, en_conv_cache2, en_res_cache2 = self.enblock2(en_x1, en_conv_cache2, en_res_cache2)
        en_x3, en_conv_cache3, en_res_cache3 = self.enblock3(en_x2, en_conv_cache3, en_res_cache3)
        en_x4, en_conv_cache4, en_res_cache4 = self.enblock4(en_x3, en_conv_cache4, en_res_cache4)
        en_x5, en_conv_cache5, en_res_cache5 = self.enblock5(en_x4, en_conv_cache5, en_res_cache5)

        bottle_arg = bottle_cache if hasattr(self.bottle, "cache_names") else bottle_cache[0]
        en_xr, bottle_cache, temporal_refine_cache = self.bottle(en_x5, bottle_arg, temporal_refine_cache)
        if not isinstance(bottle_cache, (list, tuple)):
            bottle_cache = [bottle_cache]

        de_x5, de_conv_cache5, de_res_cache5 = self.deblock5(en_xr, en_x5, de_conv_cache5, de_res_cache5)
        de_x5 = de_x5[..., : en_x4.shape[-1]]
        de_x4, de_conv_cache4, de_res_cache4 = self.deblock4(de_x5, en_x4, de_conv_cache4, de_res_cache4)
        de_x4 = de_x4[..., : en_x3.shape[-1]]
        de_x3, de_conv_cache3, de_res_cache3 = self.deblock3(de_x4, en_x3, de_conv_cache3, de_res_cache3)
        de_x3 = de_x3[..., : en_x2.shape[-1]]
        de_x2, de_conv_cache2, de_res_cache2 = self.deblock2(de_x3, en_x2, de_conv_cache2, de_res_cache2)
        de_x2 = de_x2[..., : en_x1.shape[-1]]
        de_x1, de_conv_cache1, de_res_cache1 = self.deblock1(de_x2, en_x1, de_conv_cache1, de_res_cache1)
        de_x1 = de_x1[..., : en_x0.shape[-1]]

        x_enh, m_cache = self.ccm(de_x1, x, m_cache)

        new_cache = [
            en_conv_cache1,
            en_res_cache1,
            en_conv_cache2,
            en_res_cache2,
            en_conv_cache3,
            en_res_cache3,
            en_conv_cache4,
            en_res_cache4,
            en_conv_cache5,
            en_res_cache5,
            *bottle_cache,
        ]
        if self.bottle.refine is not None:
            new_cache.append(temporal_refine_cache)
        new_cache.extend([
            de_conv_cache5,
            de_res_cache5,
            de_conv_cache4,
            de_res_cache4,
            de_conv_cache3,
            de_res_cache3,
            de_conv_cache2,
            de_res_cache2,
            de_conv_cache1,
            de_res_cache1,
            m_cache,
        ])
        return x_enh, new_cache

    def forward_flat(self, x, *cache):
        """ONNX-friendly wrapper: returns ``(enh, *new_cache)``."""
        y, new_cache = self.forward(x, list(cache))
        return (y, *new_cache)


class StreamDeepVQE_AblationONNXWrapper(nn.Module):
    def __init__(self, stream_model):
        super().__init__()
        self.stream_model = stream_model

    def forward(self, x, *cache):
        return self.stream_model.forward_flat(x, *cache)


def convert_ablation_to_stream(stream_model, model, strict=True):
    """Copy offline ablation weights into the matching stateful stream model."""
    state_dict = model.state_dict()
    new_state_dict = stream_model.state_dict()
    missing = []
    shape_mismatch = []

    for key, value in new_state_dict.items():
        candidates = (
            key,
            key.replace("Conv2d.", ""),
            key.replace("conv.Conv2d.", "conv."),
            key.replace("depthwise.Conv2d.", "depthwise."),
        )
        matched = None
        for candidate in candidates:
            if candidate in state_dict:
                matched = candidate
                break
        if matched is None:
            missing.append(key)
            continue
        if state_dict[matched].shape != value.shape:
            shape_mismatch.append((key, matched, tuple(value.shape), tuple(state_dict[matched].shape)))
            continue
        new_state_dict[key] = state_dict[matched]

    if strict and (missing or shape_mismatch):
        details = []
        if missing:
            details.append(f"missing={missing}")
        if shape_mismatch:
            details.append(f"shape_mismatch={shape_mismatch}")
        raise ValueError("Unable to convert ablation weights to stream: " + "; ".join(details))

    stream_model.load_state_dict(new_state_dict, strict=False)
    return stream_model


@torch.no_grad()
def stream_sequence(stream_model, x, cache=None):
    """Run a full ``(B, F, T, 2)`` sequence through a stateful stream model."""
    if cache is None:
        cache = stream_model.init_cache(x.shape[0], x.shape[1], x.device, x.dtype)
    outputs = []
    for frame_idx in range(x.shape[2]):
        y, cache = stream_model(x[:, :, frame_idx : frame_idx + 1, :], cache)
        outputs.append(y)
    return torch.cat(outputs, dim=2), cache


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())
