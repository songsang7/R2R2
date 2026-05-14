import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from scale_rl.agents.simbaV2.simbaV2_update import l2normalize

tfd = tfp.distributions
tfb = tfp.bijectors


class Scaler(nn.Module):
    dim: int
    init: float = 1.0
    scale: float = 1.0

    def setup(self):
        self.scaler = self.param(
            "scaler",
            nn.initializers.constant(1.0 * self.scale),
            self.dim,
        )
        self.forward_scaler = self.init / self.scale

    def __call__(self, x):
        return self.scaler * self.forward_scaler * x


class HyperDense(nn.Module):
    hidden_dim: int

    def setup(self):
        self.w = nn.Dense(
            name="hyper_dense",
            features=self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(scale=1.0, column_axis=0),
            use_bias=False,  # important!
        )

    def __call__(self, x):
        return self.w(x)


class HyperMLP(nn.Module):
    hidden_dim: int
    out_dim: int
    scaler_init: float
    scaler_scale: float
    eps: float = 1e-8

    def setup(self):
        self.w1 = HyperDense(self.hidden_dim)
        self.scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)
        self.w2 = HyperDense(self.out_dim)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.w1(x)
        x = self.scaler(x)
        # `eps` is required to prevent zero vector.
        x = nn.relu(x) + self.eps
        x = self.w2(x)
        x = l2normalize(x, axis=-1)
        return x


class HyperEmbedder(nn.Module):
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    c_shift: float

    def setup(self):
        self.w = HyperDense(self.hidden_dim)
        self.scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        new_axis = jnp.ones((x.shape[:-1] + (1,))) * self.c_shift
        x = jnp.concatenate([x, new_axis], axis=-1)
        x = l2normalize(x, axis=-1)
        x = self.w(x)
        x = self.scaler(x)
        x = l2normalize(x, axis=-1)

        return x


class HyperLERPBlock(nn.Module):
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float

    expansion: int = 4

    def setup(self):
        self.mlp = HyperMLP(
            hidden_dim=self.hidden_dim * self.expansion,
            out_dim=self.hidden_dim,
            scaler_init=self.scaler_init / math.sqrt(self.expansion),
            scaler_scale=self.scaler_scale / math.sqrt(self.expansion),
        )
        self.alpha_scaler = Scaler(
            self.hidden_dim,
            init=self.alpha_init,
            scale=self.alpha_scale,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.mlp(x)
        x = residual + self.alpha_scaler(x - residual)
        x = l2normalize(x, axis=-1)

        return x


class HyperNormalTanhPolicy(nn.Module):
    hidden_dim: int
    action_dim: int
    scaler_init: float
    scaler_scale: float
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    def setup(self):
        self.mean_w1 = HyperDense(self.hidden_dim)
        self.mean_scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)
        self.mean_w2 = HyperDense(self.action_dim)
        self.mean_bias = self.param(
            "mean_bias", nn.initializers.zeros, (self.action_dim,)
        )

        self.std_w1 = HyperDense(self.hidden_dim)
        self.std_scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)
        self.std_w2 = HyperDense(self.action_dim)
        self.std_bias = self.param(
            "std_bias", nn.initializers.zeros, (self.action_dim,)
        )

    def __call__(
        self,
        x: jnp.ndarray,
        temperature: float = 1.0,
    ) -> tfd.Distribution:
        mean = self.mean_w1(x)
        mean = self.mean_scaler(mean)
        mean = self.mean_w2(mean) + self.mean_bias

        log_std = self.std_w1(x)
        log_std = self.std_scaler(log_std)
        log_std = self.std_w2(log_std) + self.std_bias

        # normalize log-stds for stability
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1 + nn.tanh(log_std)
        )

        # N(mu, exp(log_sigma))
        dist = tfd.MultivariateNormalDiag(
            loc=mean,
            scale_diag=jnp.exp(log_std) * temperature,
        )

        # tanh(N(mu, sigma))
        dist = tfd.TransformedDistribution(distribution=dist, bijector=tfb.Tanh())

        info = {}
        return dist, info


class HyperCategoricalValue(nn.Module):
    hidden_dim: int
    num_bins: int
    min_v: float
    max_v: float
    scaler_init: float
    scaler_scale: float

    def setup(self):
        self.w1 = HyperDense(self.hidden_dim)
        self.scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)
        self.w2 = HyperDense(self.num_bins)
        self.bias = self.param("value_bias", nn.initializers.zeros, (self.num_bins,))
        self.bin_values = jnp.linspace(
            start=self.min_v, stop=self.max_v, num=self.num_bins
        ).reshape(1, -1)

    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        value = self.w1(x)
        value = self.scaler(value)
        value = self.w2(value) + self.bias

        # return log probability of bins
        log_prob = nn.log_softmax(value, axis=1)
        value = jnp.sum(jnp.exp(log_prob) * self.bin_values, axis=1)

        info = {"log_prob": log_prob}
        return value, info
