"""
SimbaV2-SPL Agent - JAX Implementation

Implements State Prediction Layer (SPL) approach with SimbaV2 architecture
and observation projection for actor/critic.
"""

import functools
from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.jax_utils.network import Network, PRNGKey
from scale_rl.agents.simbaV2.simbaV2_update import (
    l2normalize_network,
    update_target_network,
    update_temperature,
)
from scale_rl.agents.simbaV2_spl.simbaV2_spl_network import (
    HypersphereEncoder,
    HypersphereLatentDynamicsModel,
    SimbaV2SplActor,
    SimbaV2SplCritic,
    SimbaV2SplDoubleCritic,
    SimbaV2SplTemperature,
)
from scale_rl.agents.simbaV2_spl.simbaV2_spl_update import (
    update_encoder_and_dynamics,
    update_actor_spl,
    update_critic_spl,
)
from scale_rl.buffers.base_buffer import Batch


@dataclass(frozen=True)
class SimbaV2SplConfig:
    """Configuration for SimbaV2-SPL agent."""

    seed: int
    normalize_observation: bool
    normalize_reward: bool
    normalized_g_max: float

    load_only_param: bool
    load_param_key: bool
    load_observation_normalizer: bool
    load_reward_normalizer: bool

    learning_rate_init: float
    learning_rate_end: float
    learning_rate_decay_rate: float
    learning_rate_decay_step: int

    encoder_lr_init: float
    encoder_lr_end: float

    encoder_hidden_dim: int
    encoder_c_shift: float
    encoder_scaler_init: float
    encoder_scaler_scale: float

    dynamics_hidden_dim: int

    actor_num_blocks: int
    actor_hidden_dim: int
    actor_c_shift: float
    actor_scaler_init: float
    actor_scaler_scale: float
    actor_alpha_init: float
    actor_alpha_scale: float
    actor_bc_alpha: float

    critic_use_cdq: bool
    critic_num_blocks: int
    critic_hidden_dim: int
    critic_emb_dim: int
    critic_c_shift: float
    critic_num_bins: int
    critic_min_v: float
    critic_max_v: float
    critic_scaler_init: float
    critic_scaler_scale: float
    critic_alpha_init: float
    critic_alpha_scale: float

    target_tau: float

    temp_initial_value: float
    temp_target_entropy: float
    temp_target_entropy_coef: float

    gamma: float
    n_step: int


@functools.partial(
    jax.jit,
    static_argnames=(
        "observation_dim",
        "action_dim",
        "cfg",
    ),
)
def _init_simbaV2_spl_networks(
    observation_dim: int,
    action_dim: int,
    cfg: SimbaV2SplConfig,
) -> Tuple[
    PRNGKey,
    Network,
    Network,
    Network,
    Network,
    Network,
    Network,
    Network,
    Network,
]:
    """Initialize all networks for SimbaV2-SPL agent."""

    fake_observations = jnp.zeros((1, observation_dim))
    fake_z_state = jnp.zeros((1, cfg.encoder_hidden_dim))
    fake_actions = jnp.zeros((1, action_dim))

    rng = jax.random.PRNGKey(cfg.seed)
    rng, encoder_key, dynamics_key, actor_key, critic_key, temp_key = jax.random.split(
        rng, 6
    )

    encoder_lr_schedule = optax.linear_schedule(
        init_value=cfg.encoder_lr_init,
        end_value=cfg.encoder_lr_end,
        transition_steps=cfg.learning_rate_decay_step,
    )

    standard_lr_schedule = optax.linear_schedule(
        init_value=cfg.learning_rate_init,
        end_value=cfg.learning_rate_end,
        transition_steps=cfg.learning_rate_decay_step,
    )

    encoder = Network.create(
        network_def=HypersphereEncoder(
            hidden_dim=cfg.encoder_hidden_dim,
            scaler_init=cfg.encoder_scaler_init,
            scaler_scale=cfg.encoder_scaler_scale,
            c_shift=cfg.encoder_c_shift,
        ),
        network_inputs={"rngs": encoder_key, "observations": fake_observations},
        tx=optax.adam(learning_rate=encoder_lr_schedule),
    )

    target_encoder = Network.create(
        network_def=HypersphereEncoder(
            hidden_dim=cfg.encoder_hidden_dim,
            scaler_init=cfg.encoder_scaler_init,
            scaler_scale=cfg.encoder_scaler_scale,
            c_shift=cfg.encoder_c_shift,
        ),
        network_inputs={"rngs": encoder_key, "observations": fake_observations},
        tx=None,
    )

    lat_env_model = Network.create(
        network_def=HypersphereLatentDynamicsModel(
            hidden_dim=cfg.encoder_hidden_dim,
            dynamics_hidden_dim=cfg.dynamics_hidden_dim,
            action_dim=action_dim,
        ),
        network_inputs={
            "rngs": dynamics_key,
            "z_state": fake_z_state,
            "action": fake_actions,
        },
        tx=optax.adam(learning_rate=encoder_lr_schedule),
    )

    target_lat_env_model = Network.create(
        network_def=HypersphereLatentDynamicsModel(
            hidden_dim=cfg.encoder_hidden_dim,
            dynamics_hidden_dim=cfg.dynamics_hidden_dim,
            action_dim=action_dim,
        ),
        network_inputs={
            "rngs": dynamics_key,
            "z_state": fake_z_state,
            "action": fake_actions,
        },
        tx=None,
    )

    actor = Network.create(
        network_def=SimbaV2SplActor(
            num_blocks=cfg.actor_num_blocks,
            hidden_dim=cfg.actor_hidden_dim,
            action_dim=action_dim,
            c_shift=cfg.actor_c_shift,
            scaler_init=cfg.actor_scaler_init,
            scaler_scale=cfg.actor_scaler_scale,
            alpha_init=cfg.actor_alpha_init,
            alpha_scale=cfg.actor_alpha_scale,
        ),
        network_inputs={
            "rngs": actor_key,
            "observation": fake_observations,
            "z_state": fake_z_state,
        },
        tx=optax.adam(learning_rate=standard_lr_schedule),
    )

    if cfg.critic_use_cdq:
        critic_network_def = SimbaV2SplDoubleCritic(
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
            emb_dim=cfg.critic_emb_dim,
            c_shift=cfg.critic_c_shift,
            num_bins=cfg.critic_num_bins,
            min_v=cfg.critic_min_v,
            max_v=cfg.critic_max_v,
            scaler_init=cfg.critic_scaler_init,
            scaler_scale=cfg.critic_scaler_scale,
            alpha_init=cfg.critic_alpha_init,
            alpha_scale=cfg.critic_alpha_scale,
        )
    else:
        critic_network_def = SimbaV2SplCritic(
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
            emb_dim=cfg.critic_emb_dim,
            c_shift=cfg.critic_c_shift,
            num_bins=cfg.critic_num_bins,
            min_v=cfg.critic_min_v,
            max_v=cfg.critic_max_v,
            scaler_init=cfg.critic_scaler_init,
            scaler_scale=cfg.critic_scaler_scale,
            alpha_init=cfg.critic_alpha_init,
            alpha_scale=cfg.critic_alpha_scale,
        )

    critic = Network.create(
        network_def=critic_network_def,
        network_inputs={
            "rngs": critic_key,
            "observation": fake_observations,
            "z_state": fake_z_state,
            "action": fake_actions,
            "z_next_state": fake_z_state,
        },
        tx=optax.adam(learning_rate=standard_lr_schedule),
    )

    target_critic = Network.create(
        network_def=critic_network_def,
        network_inputs={
            "rngs": critic_key,
            "observation": fake_observations,
            "z_state": fake_z_state,
            "action": fake_actions,
            "z_next_state": fake_z_state,
        },
        tx=None,
    )

    temperature = Network.create(
        network_def=SimbaV2SplTemperature(cfg.temp_initial_value),
        network_inputs={"rngs": temp_key},
        tx=optax.adam(learning_rate=standard_lr_schedule),
    )

    encoder = l2normalize_network(encoder)
    target_encoder = l2normalize_network(target_encoder)
    actor = l2normalize_network(actor)
    critic = l2normalize_network(critic)
    target_critic = l2normalize_network(target_critic)

    return (
        rng,
        encoder,
        target_encoder,
        lat_env_model,
        target_lat_env_model,
        actor,
        critic,
        target_critic,
        temperature,
    )


@jax.jit
def _sample_simbaV2_spl_actions(
    rng: PRNGKey,
    encoder: Network,
    actor: Network,
    observations: jnp.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    """Sample actions from the policy."""
    rng, key = jax.random.split(rng)

    z_state = encoder(observations=observations)

    dist, _ = actor(observation=observations, z_state=z_state, temperature=temperature)
    actions = dist.sample(seed=key)

    return rng, actions


@functools.partial(
    jax.jit,
    static_argnames=(
        "gamma",
        "n_step",
        "critic_use_cdq",
        "critic_min_v",
        "critic_max_v",
        "critic_num_bins",
        "target_tau",
        "temp_target_entropy",
        "actor_bc_alpha",
    ),
)
def _update_simbaV2_spl_networks(
    rng: PRNGKey,
    encoder: Network,
    target_encoder: Network,
    lat_env_model: Network,
    target_lat_env_model: Network,
    actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    batch: Batch,
    gamma: float,
    n_step: int,
    actor_bc_alpha: float,
    critic_use_cdq: bool,
    critic_min_v: float,
    critic_max_v: float,
    critic_num_bins: int,
    target_tau: float,
    temp_target_entropy: float,
) -> Tuple[
    PRNGKey,
    Network,
    Network,
    Network,
    Network,
    Network,
    Network,
    Network,
    Network,
    Dict[str, float],
]:
    """
    Update all networks.

    Update order:
    1. Encoder + LatentDynamicsModel (MSE loss)
    2. Actor
    3. Temperature
    4. Critic
    5. All target networks (soft update)
    """
    rng, actor_key, critic_key = jax.random.split(rng, 3)

    new_encoder, new_lat_env_model, encoder_info = update_encoder_and_dynamics(
        encoder=encoder,
        lat_env_model=lat_env_model,
        batch=batch,
    )

    new_actor, actor_info = update_actor_spl(
        key=actor_key,
        actor=actor,
        critic=critic,
        target_encoder=target_encoder,
        lat_env_model=lat_env_model,
        temperature=temperature,
        batch=batch,
        use_cdq=critic_use_cdq,
        bc_alpha=actor_bc_alpha,
    )

    new_temperature, temperature_info = update_temperature(
        temperature=temperature,
        entropy=actor_info["actor/entropy"],
        target_entropy=temp_target_entropy,
    )

    new_critic, critic_info = update_critic_spl(
        key=critic_key,
        actor=new_actor,
        critic=critic,
        target_critic=target_critic,
        target_encoder=target_encoder,
        target_lat_env_model=target_lat_env_model,
        temperature=new_temperature,
        batch=batch,
        use_cdq=critic_use_cdq,
        min_v=critic_min_v,
        max_v=critic_max_v,
        num_bins=critic_num_bins,
        gamma=gamma,
        n_step=n_step,
    )

    new_target_encoder, _ = update_target_network(
        network=new_encoder,
        target_network=target_encoder,
        target_tau=target_tau,
    )
    new_target_lat_env_model, _ = update_target_network(
        network=new_lat_env_model,
        target_network=target_lat_env_model,
        target_tau=target_tau,
    )
    new_target_critic, target_critic_info = update_target_network(
        network=new_critic,
        target_network=target_critic,
        target_tau=target_tau,
    )

    info = {
        **encoder_info,
        **actor_info,
        **critic_info,
        **target_critic_info,
        **temperature_info,
    }

    return (
        rng,
        new_encoder,
        new_target_encoder,
        new_lat_env_model,
        new_target_lat_env_model,
        new_actor,
        new_critic,
        new_target_critic,
        new_temperature,
        info,
    )


class SimbaV2SplAgent(BaseAgent):
    """
    SimbaV2-SPL Agent implementing State Prediction Layer approach with observation projection.

    This agent uses:
    - HypersphereEncoder to convert observations to z_state on unit sphere
    - HypersphereLatentDynamicsModel to predict next z_state
    - SimbaV2SplActor that takes [observation, z_state] with projection
    - SimbaV2SplCritic that takes [observation, z_state, action, z_next_state] with projection
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        cfg: SimbaV2SplConfig,
    ):
        self._observation_dim = observation_space.shape[-1]
        self._action_dim = action_space.shape[-1]

        cfg["temp_target_entropy"] = cfg["temp_target_entropy_coef"] * self._action_dim

        super(SimbaV2SplAgent, self).__init__(
            observation_space,
            action_space,
            cfg,
        )

        self._cfg = SimbaV2SplConfig(**cfg)

        (
            self._rng,
            self._encoder,
            self._target_encoder,
            self._lat_env_model,
            self._target_lat_env_model,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
        ) = _init_simbaV2_spl_networks(
            self._observation_dim, self._action_dim, self._cfg
        )

    def sample_actions(
        self,
        interaction_step: int,
        prev_timestep: Dict[str, np.ndarray],
        training: bool,
    ) -> np.ndarray:
        if training:
            temperature = 1.0
        else:
            temperature = 0.0

        observations = jnp.asarray(prev_timestep["next_observation"])

        self._rng, actions = _sample_simbaV2_spl_actions(
            self._rng, self._encoder, self._actor, observations, temperature
        )
        actions = np.array(actions)

        return actions

    def update(self, update_step: int, batch: Dict[str, np.ndarray]) -> Dict:
        for key, value in batch.items():
            batch[key] = jnp.asarray(value)

        (
            self._rng,
            self._encoder,
            self._target_encoder,
            self._lat_env_model,
            self._target_lat_env_model,
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
            update_info,
        ) = _update_simbaV2_spl_networks(
            rng=self._rng,
            encoder=self._encoder,
            target_encoder=self._target_encoder,
            lat_env_model=self._lat_env_model,
            target_lat_env_model=self._target_lat_env_model,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            batch=batch,
            gamma=self._cfg.gamma,
            n_step=self._cfg.n_step,
            critic_use_cdq=self._cfg.critic_use_cdq,
            critic_min_v=self._cfg.critic_min_v,
            critic_max_v=self._cfg.critic_max_v,
            critic_num_bins=self._cfg.critic_num_bins,
            target_tau=self._cfg.target_tau,
            temp_target_entropy=self._cfg.temp_target_entropy,
            actor_bc_alpha=self._cfg.actor_bc_alpha,
        )

        for key, value in update_info.items():
            if isinstance(value, dict):
                continue
            update_info[key] = float(value)

        return update_info

    def save(self, path: str) -> None:
        self._encoder.save(path + "/encoder")
        self._target_encoder.save(path + "/target_encoder")
        self._lat_env_model.save(path + "/lat_env_model")
        self._target_lat_env_model.save(path + "/target_lat_env_model")
        self._actor.save(path + "/actor")
        self._critic.save(path + "/critic")
        self._target_critic.save(path + "/target_critic")
        self._temperature.save(path + "/temperature")

    def load(self, path: str) -> None:
        only_param = self._cfg.load_only_param
        param_key = self._cfg.load_param_key

        self._encoder = self._encoder.load(path + "/encoder", param_key, only_param)
        self._target_encoder = self._target_encoder.load(
            path + "/target_encoder", param_key, only_param
        )
        self._lat_env_model = self._lat_env_model.load(
            path + "/lat_env_model", param_key, only_param
        )
        self._target_lat_env_model = self._target_lat_env_model.load(
            path + "/target_lat_env_model", param_key, only_param
        )
        self._actor = self._actor.load(path + "/actor", param_key, only_param)
        self._critic = self._critic.load(path + "/critic", param_key, only_param)
        self._target_critic = self._target_critic.load(
            path + "/target_critic", param_key, only_param
        )
        self._temperature = self._temperature.load(
            path + "/temperature", None, only_param
        )
