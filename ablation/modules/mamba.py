"""Native PyTorch Mamba-style state space blocks.

This implementation favors a small, explicit streaming state contract over
custom CUDA kernels so ablation variants remain portable to ONNX/runtime
deployments.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NativeMambaBlock(nn.Module):
    """Causal Mamba-style block with an explicit recurrent cache.

    Input and output are shaped ``(B, T, D)``. The streaming cache is a tuple
    ``(conv_cache, ssm_state)`` where ``conv_cache`` stores the previous
    ``d_conv - 1`` projected samples and ``ssm_state`` stores per-channel SSM
    state ``(B, d_inner, d_state)``.
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        ssm_state_clip=1e4,
        ssm_param_clip=10.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.d_model * self.expand
        self.dt_max = float(dt_max)
        self.dt_min = float(dt_init_floor)
        self.ssm_state_clip = float(ssm_state_clip)
        self.ssm_param_clip = float(ssm_param_clip)
        if self.d_state < 1:
            raise ValueError("d_state must be >= 1")
        if self.d_conv < 1:
            raise ValueError("d_conv must be >= 1")
        if dt_rank == "auto":
            dt_rank = math.ceil(self.d_model / 16)
        self.dt_rank = int(dt_rank)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            bias=True,
        )
        self.activation = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        a = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)

    def init_cache(self, batch_size, device=None, dtype=None):
        conv_len = max(0, self.d_conv - 1)
        conv_cache = torch.zeros(
            int(batch_size),
            self.d_inner,
            conv_len,
            device=device,
            dtype=dtype,
        )
        ssm_state = torch.zeros(
            int(batch_size),
            self.d_inner,
            self.d_state,
            device=device,
            dtype=dtype,
        )
        return conv_cache, ssm_state

    def _sanitize(self, tensor):
        return torch.nan_to_num(
            tensor,
            nan=0.0,
            posinf=self.ssm_state_clip,
            neginf=-self.ssm_state_clip,
        ).clamp(min=-self.ssm_state_clip, max=self.ssm_state_clip)

    def _ssm_step(self, x, ssm_state):
        x_proj = self.x_proj(x)
        dt, b, c = torch.split(
            x_proj,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        x_float = x.float()
        state_float = ssm_state.float()
        dt = F.softplus(self.dt_proj(dt.float())).clamp(min=self.dt_min, max=self.dt_max)
        b = b.float().clamp(min=-self.ssm_param_clip, max=self.ssm_param_clip)
        c = c.float().clamp(min=-self.ssm_param_clip, max=self.ssm_param_clip)
        a = -torch.exp(self.A_log.float()).clamp(max=1e4)
        d_a = torch.exp((dt.unsqueeze(-1) * a.unsqueeze(0)).clamp(min=-60.0, max=0.0))
        d_b = dt.unsqueeze(-1) * b.unsqueeze(1)
        state_float = state_float * d_a + x_float.unsqueeze(-1) * d_b
        state_float = self._sanitize(state_float)
        y = torch.sum(state_float * c.unsqueeze(1), dim=-1)
        y = y + x_float * self.D.float()
        y = self._sanitize(y)
        return y.to(dtype=x.dtype), state_float

    def _causal_conv(self, x, conv_cache=None):
        x_t = x.transpose(1, 2)
        if conv_cache is None:
            x_t = F.pad(x_t, (self.d_conv - 1, 0))
        else:
            x_t = torch.cat([conv_cache, x_t], dim=2)
        x_t = self.conv1d(x_t)
        return x_t.transpose(1, 2)

    def forward(self, x, cache=None):
        if x.shape[-1] != self.d_model:
            raise ValueError(f"NativeMambaBlock expected {self.d_model} features, got {x.shape[-1]}")

        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)
        conv_cache_in = None if cache is None else cache[0]
        projected = x_part.transpose(1, 2)
        x_part = self.activation(self._causal_conv(x_part, conv_cache_in))

        if cache is None:
            ssm_state = torch.zeros(
                x.shape[0],
                self.d_inner,
                self.d_state,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            _, ssm_state = cache

        outputs = []
        for t in range(x_part.shape[1]):
            y_t, ssm_state = self._ssm_step(x_part[:, t], ssm_state)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        y = y * self.activation(z)
        y = self.out_proj(y)
        y = self._sanitize(y)

        conv_len = max(0, self.d_conv - 1)
        if conv_len > 0:
            if conv_cache_in is not None:
                projected = torch.cat([conv_cache_in, projected], dim=2)
            conv_cache = projected[:, :, -conv_len:].contiguous()
        else:
            conv_cache = x.new_zeros(x.shape[0], self.d_inner, 0)
        return y, (conv_cache, ssm_state)

    def step(self, x, cache):
        """Run one frame shaped ``(B, D)`` and return ``(B, D), cache``."""
        if x.shape[-1] != self.d_model:
            raise ValueError(f"NativeMambaBlock expected {self.d_model} features, got {x.shape[-1]}")
        conv_cache, ssm_state = cache
        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)
        conv_input = torch.cat([conv_cache, x_part.unsqueeze(-1)], dim=2)
        weight = self.conv1d.weight.squeeze(1)
        x_conv = torch.sum(conv_input * weight.unsqueeze(0), dim=2)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x_conv = self.activation(x_conv)
        conv_len = max(0, self.d_conv - 1)
        if conv_len > 0:
            conv_cache = conv_input[:, :, -conv_len:].contiguous()
        else:
            conv_cache = conv_input[:, :, :0].contiguous()

        y, ssm_state = self._ssm_step(x_conv, ssm_state)
        y = y * self.activation(z)
        y = self.out_proj(y)
        y = self._sanitize(y)
        return y, (conv_cache, ssm_state)


class ResidualMambaBlock(nn.Module):
    """LayerNorm + Mamba block with residual output."""

    def __init__(self, d_model, **mamba_kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(int(d_model))
        self.mamba = NativeMambaBlock(d_model, **mamba_kwargs)

    @property
    def d_inner(self):
        return self.mamba.d_inner

    @property
    def d_state(self):
        return self.mamba.d_state

    @property
    def d_conv(self):
        return self.mamba.d_conv

    def init_cache(self, batch_size, device=None, dtype=None):
        return self.mamba.init_cache(batch_size, device=device, dtype=dtype)

    def forward(self, x, cache=None):
        y, cache = self.mamba(self.norm(x), cache)
        return x + y, cache

    def step(self, x, cache):
        y, cache = self.mamba.step(self.norm(x), cache)
        return x + y, cache


class MambaStack(nn.Module):
    """Stack of residual Mamba blocks with flat cache helpers."""

    def __init__(
        self,
        d_model,
        num_blocks=2,
        hidden_size=None,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_blocks = int(num_blocks)
        self.hidden_size = int(hidden_size) if hidden_size is not None else self.d_model
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")

        self.in_proj = (
            nn.Linear(self.d_model, self.hidden_size)
            if self.hidden_size != self.d_model
            else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [
                ResidualMambaBlock(
                    self.hidden_size,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_size)
        self.out_proj = (
            nn.Linear(self.hidden_size, self.d_model)
            if self.hidden_size != self.d_model
            else nn.Identity()
        )

    def cache_names(self, prefix="mamba"):
        names = []
        for idx in range(self.num_blocks):
            names.append(f"{prefix}{idx + 1}_conv_cache")
            names.append(f"{prefix}{idx + 1}_ssm_state")
        return tuple(names)

    def init_cache(self, batch_size, device=None, dtype=None):
        cache = []
        for block in self.blocks:
            block_cache = block.init_cache(batch_size, device=device, dtype=dtype)
            cache.extend(block_cache)
        return tuple(cache)

    def _pack_cache(self, flat_cache):
        if len(flat_cache) != self.num_blocks * 2:
            raise ValueError(f"Expected {self.num_blocks * 2} Mamba cache tensors, got {len(flat_cache)}")
        return [
            (flat_cache[2 * idx], flat_cache[2 * idx + 1])
            for idx in range(self.num_blocks)
        ]

    def _unpack_cache(self, block_caches):
        flat = []
        for conv_cache, ssm_state in block_caches:
            flat.extend([conv_cache, ssm_state])
        return tuple(flat)

    def forward(self, x, cache=None):
        y = self.in_proj(x)
        block_caches = (
            [None] * self.num_blocks
            if cache is None
            else self._pack_cache(cache)
        )
        new_caches = []
        for block, block_cache in zip(self.blocks, block_caches):
            y, block_cache = block(y, block_cache)
            new_caches.append(block_cache)
        y = self.out_proj(self.norm(y))
        return y, self._unpack_cache(new_caches)

    def step(self, x, cache):
        y = self.in_proj(x)
        new_caches = []
        for block, block_cache in zip(self.blocks, self._pack_cache(cache)):
            y, block_cache = block.step(y, block_cache)
            new_caches.append(block_cache)
        y = self.out_proj(self.norm(y))
        return y, self._unpack_cache(new_caches)
