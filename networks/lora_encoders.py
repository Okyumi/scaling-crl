"""
LoRA-style factorized variants of the SA_encoder, G_encoder, and Actor networks.

These modules are IDENTICAL to their full-rank counterparts in train.py except
that the Dense(m->m) layers inside the residual blocks are factorized into
Dense(m->r) -> Dense(r->m) using LoRADense. The network still operates at the
full width m everywhere — only the weight matrices are constrained to be rank-r.

Key invariants preserved from train.py:
- Initial Dense(input->m) layer is full-rank
- Final Dense(m->64) / Dense(m->action_size) layers are full-rank
- LayerNorm and activation operate at dimension m
- Skip connections add tensors of dimension m
- Same attributes (network_width, network_depth, skip_connections, use_relu, etc.)
"""

import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import variance_scaling

from networks.lora_layers import LoRADense


# Match the existing codebase's initialization
lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
bias_init = nn.initializers.zeros


def lora_residual_block(x, width, rank, normalize, activation):
    """Same as residual_block in train.py but Dense(width,width) layers are
    factorized into Dense(width,rank)->Dense(rank,width) via LoRADense.

    The activations stay at dimension `width` throughout. Only the weight
    matrices are constrained to be rank-`rank`.
    """
    identity = x
    x = LoRADense(features=width, rank=rank)(x)
    x = normalize(x)
    x = activation(x)
    x = LoRADense(features=width, rank=rank)(x)
    x = normalize(x)
    x = activation(x)
    x = LoRADense(features=width, rank=rank)(x)
    x = normalize(x)
    x = activation(x)
    x = LoRADense(features=width, rank=rank)(x)
    x = normalize(x)
    x = activation(x)
    x = x + identity
    return x


class LoraSAEncoder(nn.Module):
    """LoRA-style factorized State-Action encoder for the critic.

    Identical to SA_encoder in train.py except the Dense(m->m) layers inside
    residual blocks are factorized via LoRADense. All other layers (initial
    Dense, final Dense, LayerNorm, activation) are unchanged.
    """
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    low_rank_dim: int = 64

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        x = jnp.concatenate([s, a], axis=-1)
        #Initial layer (full-rank)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        #Residual blocks (LoRA-factorized Dense layers)
        for i in range(self.network_depth // 4):
            x = lora_residual_block(x, self.network_width, self.low_rank_dim, normalize, activation)
        #Final layer (full-rank)
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class LoraGEncoder(nn.Module):
    """LoRA-style factorized Goal encoder for the critic.

    Identical to G_encoder in train.py except the Dense(m->m) layers inside
    residual blocks are factorized via LoRADense. All other layers (initial
    Dense, final Dense, LayerNorm, activation) are unchanged.
    """
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    low_rank_dim: int = 64

    @nn.compact
    def __call__(self, g: jnp.ndarray):

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        x = g
        #Initial layer (full-rank)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        #Residual blocks (LoRA-factorized Dense layers)
        for i in range(self.network_depth // 4):
            x = lora_residual_block(x, self.network_width, self.low_rank_dim, normalize, activation)
        #Final layer (full-rank)
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class LoraActor(nn.Module):
    """LoRA-style factorized Actor network (policy).

    Identical to Actor in train.py except the Dense(m->m) layers inside
    residual blocks are factorized via LoRADense. All other layers (initial
    Dense, final mean/log_std Dense heads, LayerNorm, activation) are unchanged.
    """
    action_size: int
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: int = 0
    low_rank_dim: int = 64
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        #Initial layer (full-rank)
        x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = activation(x)
        #Residual blocks (LoRA-factorized Dense layers)
        for i in range(self.network_depth // 4):
            x = lora_residual_block(x, self.network_width, self.low_rank_dim, normalize, activation)
        #Final layer (full-rank)
        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std
