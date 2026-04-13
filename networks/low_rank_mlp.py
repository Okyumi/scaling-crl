"""
Low-rank residual MLP in JAX/Flax.

Ported from PyTorch implementation in low_rank_res.py.
Architecture: V(d→r) → low-rank residual blocks(r→r) → U(r→m) → output(m→k)

The low-rank residual blocks operate in a bottleneck dimension r << m, using
the same Dense + LayerNorm + Swish pattern with skip connections every 4 layers
as the existing full-rank residual blocks in train.py.

Initialization uses epsilon-scaled orthogonal weights (a simplified version
of the SVD-based gradient initialization in the PyTorch reference, which is
complex to implement in JAX's functional paradigm).
"""

import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import variance_scaling, orthogonal


# Match the existing codebase's initialization
lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
bias_init = nn.initializers.zeros

# Epsilon-scaled orthogonal init (simplified version of SVD-based init)
def eps_orthogonal(eps=0.1):
    """Orthogonal initialization scaled by epsilon."""
    return orthogonal(scale=eps)


def low_rank_residual_block(x, width, normalize, activation):
    """
    A single residual block of 4 layers operating in the low-rank space.
    Mirrors the structure of residual_block() in train.py but operates
    at the bottleneck dimension.

    Args:
        x: input tensor of shape (..., width) where width = low_rank_dim
        width: the low-rank dimension r
        normalize: normalization function (LayerNorm)
        activation: activation function (swish or relu)

    Returns:
        output tensor of shape (..., width) with skip connection added
    """
    identity = x
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = x + identity
    return x


class LowRankMLP(nn.Module):
    """
    Low-rank bottleneck MLP with residual connections.

    Architecture:
        1. V projection:   input_dim → low_rank_dim
        2. LayerNorm + activation
        3. Residual blocks: low_rank_dim → low_rank_dim (skip every 4 layers)
        4. U expansion:    low_rank_dim → hidden_dim
        5. LayerNorm + activation
        6. Output layer:   hidden_dim → output_dim

    This mirrors the full-rank architecture in train.py (SA_encoder, G_encoder)
    but operates the residual stack in a lower-dimensional bottleneck space.

    Attributes:
        hidden_dim: width of the full-rank space (m), used after U expansion
        low_rank_dim: bottleneck dimension (r) for the residual blocks
        output_dim: final output dimension (k)
        depth: total number of hidden layers (must be divisible by 4)
        use_relu: if True, use ReLU; otherwise use Swish
        eps: epsilon for scaled orthogonal init of V and U projections
    """
    hidden_dim: int = 256
    low_rank_dim: int = 64
    output_dim: int = 64
    depth: int = 4
    use_relu: int = 0
    eps: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        normalize = lambda x: nn.LayerNorm()(x)
        activation = nn.relu if self.use_relu else nn.swish

        # V projection: input_dim → low_rank_dim
        # Uses eps-scaled orthogonal init for the projection into low-rank space
        x = nn.Dense(
            self.low_rank_dim,
            kernel_init=eps_orthogonal(self.eps),
            bias_init=bias_init,
        )(x)
        x = normalize(x)
        x = activation(x)

        # Low-rank residual blocks (operate at dimension r)
        for i in range(self.depth // 4):
            x = low_rank_residual_block(x, self.low_rank_dim, normalize, activation)

        # U expansion: low_rank_dim → hidden_dim
        # Uses eps-scaled orthogonal init for the projection back to full-rank space
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=eps_orthogonal(self.eps),
            bias_init=bias_init,
        )(x)
        x = normalize(x)
        x = activation(x)

        # Output layer: hidden_dim → output_dim
        x = nn.Dense(
            self.output_dim,
            kernel_init=lecun_uniform,
            bias_init=bias_init,
        )(x)

        return x
