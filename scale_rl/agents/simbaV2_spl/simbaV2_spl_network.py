"""
SimbaV2-SPL Networks - JAX/Flax Implementation

Networks implementing State Prediction Layer (SPL) approach with observation projection:
- HypersphereEncoder: Encodes observations to hypersphere representation
- HypersphereLatentDynamicsModel: Predicts next state embedding on hypersphere
- ObsProjector: 1-layer MLP for observation projection
- SimbaV2SplActor: Takes [observation, z_state] with projection
- SimbaV2SplCritic: Takes [observation, z_state, action, z_next_state] with projection
- SimbaV2SplDoubleCritic: Double Q-network for Clipped Double Q-learning
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from tensorflow_probability.substrates import jax as tfp

from scale_rl.agents.simbaV2.simbaV2_layer import (
    HyperEmbedder,
    HyperLERPBlock,
    HyperNormalTanhPolicy,
    HyperCategoricalValue,
)
from scale_rl.agents.simbaV2.simbaV2_update import l2normalize

tfd = tfp.distributions
tfb = tfp.bijectors


class HypersphereEncoder(nn.Module):
    """
    State encoder using HyperEmbedder.

    Encodes observations to hypersphere (unit sphere) representation.
    Output is L2 normalized and has hidden_dim dimensions.

    Attributes:
        hidden_dim: Output dimension (z_state dimension)
        scaler_init: Initial scaling factor
        scaler_scale: Scaling parameter
        c_shift: Constant shift for embedding
    """

    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    c_shift: float

    def setup(self):
        self.embedder = HyperEmbedder(
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            c_shift=self.c_shift,
        )

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        """
        Encode observations to hypersphere.

        Args:
            observations: Observation tensor of shape (batch, obs_dim)

        Returns:
            z_state: Encoded state on unit sphere, shape (batch, hidden_dim)
        """
        return self.embedder(observations)


class HypersphereLatentDynamicsModel(nn.Module):
    """
    Predicts next state embedding on hypersphere.

    Uses simple MLP with ELU activations and L2 normalization at the end.
    Input z_state is already on unit sphere.

    Attributes:
        hidden_dim: Encoder output dimension (z_state dimension)
        dynamics_hidden_dim: Internal MLP hidden dimension
        action_dim: Action dimension
    """

    hidden_dim: int
    dynamics_hidden_dim: int
    action_dim: int

    def setup(self):
        self.fc1 = nn.Dense(self.dynamics_hidden_dim)
        self.fc2 = nn.Dense(self.dynamics_hidden_dim)
        self.fc3 = nn.Dense(self.hidden_dim)

    def __call__(
        self,
        z_state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Predict next state embedding.

        Args:
            z_state: Current state embedding on unit sphere, shape (batch, hidden_dim)
            action: Action tensor, shape (batch, action_dim)

        Returns:
            z_next_pred: Predicted next state on unit sphere, shape (batch, hidden_dim)
        """
        x = jnp.concatenate([z_state, action], axis=-1)
        x = jax.nn.elu(self.fc1(x))
        x = jax.nn.elu(self.fc2(x))
        x = self.fc3(x)
        return l2normalize(x, axis=-1)


class ObsProjector(nn.Module):
    """
    1-layer Linear projection with L2 normalization.

    Projects observation (or observation+action) to hidden_dim,
    then L2 normalizes to match z_state's unit norm scale.

    Attributes:
        hidden_dim: Output dimension
    """

    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(x)
        x = l2normalize(x, axis=-1)
        return x


class SimbaV2SplActor(nn.Module):
    """
    SimbaV2-SPL Actor that takes [observation, z_state] with projection.

    Architecture:
        observation -> ObsProjector -> proj_obs
        [proj_obs, z_state] -> HyperEmbedder -> [HyperLERPBlock] * num_blocks
            -> HyperNormalTanhPolicy -> distribution

    Attributes:
        num_blocks: Number of LERP blocks
        hidden_dim: Hidden dimension (same as z_state dimension)
        action_dim: Action dimension
        scaler_init: Initial scaling factor for LERP blocks
        scaler_scale: Scaling parameter for LERP blocks
        alpha_init: Initial alpha for LERP interpolation
        alpha_scale: Alpha scaling parameter
        c_shift: Constant shift for HyperEmbedder
    """

    num_blocks: int
    hidden_dim: int
    action_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    c_shift: float

    def setup(self):
        self.obs_proj = ObsProjector(hidden_dim=self.hidden_dim)

        self.embedder = HyperEmbedder(
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            c_shift=self.c_shift,
        )

        self.encoder = nn.Sequential(
            [
                HyperLERPBlock(
                    hidden_dim=self.hidden_dim,
                    scaler_init=self.scaler_init,
                    scaler_scale=self.scaler_scale,
                    alpha_init=self.alpha_init,
                    alpha_scale=self.alpha_scale,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.predictor = HyperNormalTanhPolicy(
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def __call__(
        self,
        observation: jnp.ndarray,
        z_state: jnp.ndarray,
        temperature: float = 1.0,
    ) -> tfd.Distribution:
        """
        Forward pass.

        Args:
            observation: Raw observation, shape (batch, obs_dim)
            z_state: Encoded state on unit sphere, shape (batch, hidden_dim)
            temperature: Temperature for exploration (1.0 for training, 0.0 for eval)

        Returns:
            dist: TanhNormal distribution for actions
            info: Additional info dictionary
        """
        obs_proj = self.obs_proj(observation)
        x = jnp.concatenate([obs_proj, z_state], axis=-1)
        x = self.embedder(x)
        x = self.encoder(x)
        dist, info = self.predictor(x, temperature)
        return dist, info


class SimbaV2SplCritic(nn.Module):
    """
    SimbaV2-SPL Critic that takes [observation, z_state, action, z_next_state].

    Architecture:
        [observation, action] -> ObsProjector -> proj_obs_action
        [proj_obs_action, z_state, z_next_state] -> HyperEmbedder -> [HyperLERPBlock] * num_blocks
            -> HyperCategoricalValue -> (Q, info)

    Attributes:
        num_blocks: Number of LERP blocks
        hidden_dim: Internal hidden dimension
        emb_dim: Encoder output dimension (z_state dimension)
        scaler_init: Initial scaling factor
        scaler_scale: Scaling parameter
        alpha_init: Initial alpha for LERP interpolation
        alpha_scale: Alpha scaling parameter
        c_shift: Constant shift for embedding
        num_bins: Number of bins for categorical distribution
        min_v: Minimum value for categorical distribution
        max_v: Maximum value for categorical distribution
    """

    num_blocks: int
    hidden_dim: int
    emb_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    c_shift: float
    num_bins: int
    min_v: float
    max_v: float

    def setup(self):
        self.obs_action_proj = ObsProjector(hidden_dim=self.hidden_dim)

        self.embedder = HyperEmbedder(
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            c_shift=self.c_shift,
        )
        self.encoder = nn.Sequential(
            [
                HyperLERPBlock(
                    hidden_dim=self.hidden_dim,
                    scaler_init=self.scaler_init,
                    scaler_scale=self.scaler_scale,
                    alpha_init=self.alpha_init,
                    alpha_scale=self.alpha_scale,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.predictor = HyperCategoricalValue(
            hidden_dim=self.hidden_dim,
            num_bins=self.num_bins,
            min_v=self.min_v,
            max_v=self.max_v,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def __call__(
        self,
        observation: jnp.ndarray,
        z_state: jnp.ndarray,
        action: jnp.ndarray,
        z_next_state: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            observation: Raw observation, shape (batch, obs_dim)
            z_state: Current state embedding on unit sphere, shape (batch, emb_dim)
            action: Action tensor, shape (batch, action_dim)
            z_next_state: Next state embedding on unit sphere, shape (batch, emb_dim)

        Returns:
            q: Q-value (expected value from categorical distribution)
            info: Dictionary containing log_probs for categorical TD loss
        """
        obs_action = jnp.concatenate([observation, action], axis=-1)
        obs_action_proj = self.obs_action_proj(obs_action)

        x = jnp.concatenate([obs_action_proj, z_state, z_next_state], axis=-1)
        x = self.embedder(x)
        x = self.encoder(x)
        q, info = self.predictor(x)
        return q, info


class SimbaV2SplDoubleCritic(nn.Module):
    """
    Vectorized Double-Q for Clipped Double Q-learning with observation projection.
    https://arxiv.org/pdf/1802.09477v3

    Maintains two independent critic networks and uses the minimum Q-value
    to prevent overestimation bias.
    """

    num_blocks: int
    hidden_dim: int
    emb_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    c_shift: float
    num_bins: int
    min_v: float
    max_v: float

    num_qs: int = 2

    @nn.compact
    def __call__(
        self,
        observation: jnp.ndarray,
        z_state: jnp.ndarray,
        action: jnp.ndarray,
        z_next_state: jnp.ndarray,
    ) -> jnp.ndarray:
        VmapCritic = nn.vmap(
            SimbaV2SplCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )

        qs, infos = VmapCritic(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            emb_dim=self.emb_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
            c_shift=self.c_shift,
            num_bins=self.num_bins,
            min_v=self.min_v,
            max_v=self.max_v,
        )(observation, z_state, action, z_next_state)

        return qs, infos


class SimbaV2SplTemperature(nn.Module):
    """Learnable temperature parameter for SAC entropy regularization."""

    initial_value: float = 0.01

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            name="log_temp",
            init_fn=lambda key: jnp.full(
                shape=(), fill_value=jnp.log(self.initial_value)
            ),
        )
        return jnp.exp(log_temp)
