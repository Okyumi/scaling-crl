"""
Low-rank variants of the SA_encoder, G_encoder, and Actor networks.

These modules mirror the architecture of their full-rank counterparts in
train.py, but replace the standard residual MLP trunk with the LowRankMLP
bottleneck architecture from low_rank_mlp.py.

When use_low_rank=False (default), the original full-rank networks in train.py
are used — these modules are only instantiated when use_low_rank=True.
"""

import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import variance_scaling

from networks.low_rank_mlp import LowRankMLP


# Match the existing codebase's initialization
lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
bias_init = nn.initializers.zeros


class LowRankSAEncoder(nn.Module):
    """
    Low-rank State-Action encoder for the critic.

    Mirrors SA_encoder in train.py but uses the LowRankMLP bottleneck:
        concat(s, a) → LowRankMLP → 64-dim representation

    The LowRankMLP internally does:
        V(input→r) → residual blocks at dim r → U(r→hidden) → output(hidden→64)

    Attributes:
        network_width: full-rank hidden dimension m (default 256)
        network_depth: total depth in layers (must be divisible by 4)
        low_rank_dim: bottleneck dimension r
        use_relu: if True, use ReLU; else Swish
        eps: epsilon for scaled orthogonal init of V/U projections
    """
    network_width: int = 256
    network_depth: int = 4
    low_rank_dim: int = 64
    use_relu: int = 0
    eps: float = 0.1

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([s, a], axis=-1)
        x = LowRankMLP(
            hidden_dim=self.network_width,
            low_rank_dim=self.low_rank_dim,
            output_dim=64,  # representation dimension, matches train.py
            depth=self.network_depth,
            use_relu=self.use_relu,
            eps=self.eps,
        )(x)
        return x


class LowRankGEncoder(nn.Module):
    """
    Low-rank Goal encoder for the critic.

    Mirrors G_encoder in train.py but uses the LowRankMLP bottleneck:
        goal → LowRankMLP → 64-dim representation

    Attributes:
        network_width: full-rank hidden dimension m (default 256)
        network_depth: total depth in layers (must be divisible by 4)
        low_rank_dim: bottleneck dimension r
        use_relu: if True, use ReLU; else Swish
        eps: epsilon for scaled orthogonal init of V/U projections
    """
    network_width: int = 256
    network_depth: int = 4
    low_rank_dim: int = 64
    use_relu: int = 0
    eps: float = 0.1

    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        x = LowRankMLP(
            hidden_dim=self.network_width,
            low_rank_dim=self.low_rank_dim,
            output_dim=64,  # representation dimension, matches train.py
            depth=self.network_depth,
            use_relu=self.use_relu,
            eps=self.eps,
        )(g)
        return x


class LowRankActor(nn.Module):
    """
    Low-rank Actor network (policy).

    Mirrors Actor in train.py but uses the LowRankMLP bottleneck:
        concat(state, goal) → LowRankMLP(→hidden_dim) → mean, log_std

    Note: The LowRankMLP here outputs hidden_dim features (not 64), and then
    separate Dense heads produce mean and log_std for the action distribution.

    Attributes:
        action_size: dimension of the action space
        network_width: full-rank hidden dimension m (default 256)
        network_depth: total depth in layers (must be divisible by 4)
        low_rank_dim: bottleneck dimension r
        use_relu: if True, use ReLU; else Swish
        eps: epsilon for scaled orthogonal init of V/U projections
    """
    action_size: int = 8
    network_width: int = 256
    network_depth: int = 4
    low_rank_dim: int = 64
    use_relu: int = 0
    eps: float = 0.1
    LOG_STD_MAX: float = 2.0
    LOG_STD_MIN: float = -5.0

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # Use LowRankMLP with output_dim = network_width to produce features
        # Then apply separate action heads (matching Actor in train.py)
        x = LowRankMLP(
            hidden_dim=self.network_width,
            low_rank_dim=self.low_rank_dim,
            output_dim=self.network_width,  # output features, not final actions
            depth=self.network_depth,
            use_relu=self.use_relu,
            eps=self.eps,
        )(x)

        # Action heads (same as Actor in train.py)
        mean = nn.Dense(
            self.action_size,
            kernel_init=lecun_uniform,
            bias_init=bias_init,
        )(x)
        log_std = nn.Dense(
            self.action_size,
            kernel_init=lecun_uniform,
            bias_init=bias_init,
        )(x)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

        return mean, log_std
