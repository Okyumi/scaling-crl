"""
LoRA-style factorized Dense layer for JAX/Flax.

Replaces a single Dense(m, m) with Dense(m, r) -> Dense(r, m), where r << m.
The network still operates at the full width m everywhere — only the weight
matrices are constrained to be rank-r. This is fundamentally different from
a narrow bottleneck that reduces the hidden dimension.
"""

import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import variance_scaling


# Match the existing codebase's initialization (correctly spelled in new files)
lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
bias_init = nn.initializers.zeros


class LoRADense(nn.Module):
    """LoRA-style factorized Dense layer: Dense(features_in, r) -> Dense(r, features_out).

    Replaces a single Dense(features_in, features_out) with two smaller Dense layers.
    The network still operates at the full width — only the weight matrix is low-rank.

    Attributes:
        features: output features (= m, the full width)
        rank: low-rank dimension r
    """
    features: int
    rank: int

    @nn.compact
    def __call__(self, x):
        # Down-projection: m -> r (no bias, matching LoRA convention)
        x = nn.Dense(self.rank, use_bias=False, kernel_init=lecun_uniform)(x)
        # Up-projection: r -> m (with bias)
        x = nn.Dense(self.features, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x
