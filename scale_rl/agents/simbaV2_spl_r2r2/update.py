"""
SimbaV2-SPL Update Functions

Key differences from SimbaV2-SPL:
- Actor receives [observation, z_state] with projection
- Critic receives [observation, z_state, action, z_next_state] with projection
- Additional update for encoder and latent dynamics model (MSE loss)
"""

from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp
import optax

from scale_rl.agents.jax_utils.network import Network, PRNGKey
from scale_rl.agents.simbaV2.simbaV2_update import (
    l2normalize_network,
    categorical_td_loss,
)
from scale_rl.buffers import Batch


def update_encoder_and_dynamics(
    encoder: Network,
    lat_env_model: Network,
    batch: Batch,
    coef_var: float,
    coef_spl: float,
    coef_rr: float,
) -> Tuple[Network, Network, Dict[str, float]]:
    """
    Update encoder and latent dynamics model with MSE loss.

    The dynamics model is trained to predict the next state embedding:
        loss = ||z_next_pred - z_next_target||^2

    where:
        - z_next_target = encoder(next_obs) (stop gradient, using current encoder per TD7)
        - z_s = encoder(obs)
        - z_next_pred = lat_env_model(z_s, action)

    Args:
        encoder: Encoder network
        lat_env_model: Latent dynamics model network
        batch: Batch of transitions

    Returns:
        new_encoder: Updated encoder
        new_lat_env_model: Updated latent dynamics model
        info: Dictionary containing loss information
    """
    z_next_target = jax.lax.stop_gradient(
        encoder(observations=batch["next_observation"])
    )

    def encoder_loss_fn(
        encoder_params: flax.core.FrozenDict[str, Any],
        lat_env_model_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        z_s = encoder.apply(
            variables={"params": encoder_params},
            observations=batch["observation"],
        )

        z_next_pred = lat_env_model.apply(
            variables={"params": lat_env_model_params},
            z_state=z_s,
            action=batch["action"],
        )

        # --- R2R2 ---

        # A. Variance Loss (loss_var)
        # PyTorch std는 기본적으로 unbiased (ddof=1)이므로 맞춤
        d = z_s.shape[-1]
        # target_std = 1.0 / jnp.sqrt(d)
        target_std = 1.0  # 고정 (like original VICReg)
        std_z = jnp.std(z_s, axis=0, ddof=1)
        loss_var = jax.nn.relu(target_std - std_z).mean()

        # B. SPL Loss (loss_spl) - 기존 MSE Loss
        loss_spl = jnp.square(z_next_pred - z_next_target).mean()

        # C. Redundancy Reduction Loss (loss_rr)
        batch_size, num_feats = z_s.shape
        corr_mat = (z_s.T @ z_s) / (batch_size - 1)
        
        # 대각 성분을 0으로 만든 off-diagonal 행렬 계산
        off_diag = corr_mat - jnp.diag(jnp.diag(corr_mat))
        
        # Off-diagonal 요소들의 제곱 합 평균
        loss_rr = jnp.sum(jnp.square(off_diag)) / (num_feats * (num_feats - 1))

        # Total Loss
        total_loss = coef_var * loss_var + coef_spl * loss_spl + coef_rr * loss_rr
        
        # --- R2R2 Loss Calculation End ---

        info = {
            "encoder/loss": total_loss,
            "encoder/z_norm": jnp.linalg.norm(z_s, axis=-1).mean(),
        }
        return total_loss, info

    def combined_loss_fn(
        params: Dict[str, flax.core.FrozenDict[str, Any]],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        return encoder_loss_fn(params["encoder"], params["lat_env_model"])

    combined_params = {
        "encoder": encoder.params,
        "lat_env_model": lat_env_model.params,
    }
    grads, info = jax.grad(combined_loss_fn, has_aux=True)(combined_params)

    encoder_updates, new_encoder_opt_state = encoder.tx.update(
        grads["encoder"], encoder.opt_state, encoder.params
    )
    new_encoder_params = optax.apply_updates(encoder.params, encoder_updates)
    new_encoder = encoder.replace(
        params=new_encoder_params,
        opt_state=new_encoder_opt_state,
        update_step=encoder.update_step + 1,
    )

    lat_env_model_updates, new_lat_env_model_opt_state = lat_env_model.tx.update(
        grads["lat_env_model"], lat_env_model.opt_state, lat_env_model.params
    )
    new_lat_env_model_params = optax.apply_updates(
        lat_env_model.params, lat_env_model_updates
    )
    new_lat_env_model = lat_env_model.replace(
        params=new_lat_env_model_params,
        opt_state=new_lat_env_model_opt_state,
        update_step=lat_env_model.update_step + 1,
    )

    new_encoder = l2normalize_network(new_encoder)

    return new_encoder, new_lat_env_model, info


def update_actor_spl(
    key: PRNGKey,
    actor: Network,
    critic: Network,
    target_encoder: Network,
    lat_env_model: Network,
    temperature: Network,
    batch: Batch,
    use_cdq: bool,
    bc_alpha: float,
) -> Tuple[Network, Dict[str, float]]:
    """
    Update actor network.

    Key difference: Actor takes [observation, z_state] with projection.

    Args:
        key: Random key
        actor: Actor network
        critic: Critic network
        target_encoder: Target encoder network (for computing z_state)
        lat_env_model: Latent dynamics model (for computing z_next_state)
        temperature: Temperature network
        batch: Batch of transitions
        use_cdq: Whether to use Clipped Double Q-learning
        bc_alpha: Behavior cloning regularization coefficient

    Returns:
        new_actor: Updated actor
        info: Dictionary containing loss information
    """
    z_s = jax.lax.stop_gradient(target_encoder(observations=batch["observation"]))

    def actor_loss_fn(
        actor_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        dist, _ = actor.apply(
            variables={"params": actor_params},
            observation=batch["observation"],
            z_state=z_s,
        )
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)

        z_next_s = lat_env_model(z_state=z_s, action=actions)

        if use_cdq:
            qs, _ = critic(
                observation=batch["observation"],
                z_state=z_s,
                action=actions,
                z_next_state=z_next_s,
            )
            q = jnp.minimum(qs[0], qs[1])
        else:
            q, _ = critic(
                observation=batch["observation"],
                z_state=z_s,
                action=actions,
                z_next_state=z_next_s,
            )

        actor_loss = (log_probs * temperature() - q).mean()

        if bc_alpha > 0:
            q_abs = jax.lax.stop_gradient(jnp.abs(q).mean())
            bc_loss = jnp.square(actions - batch["action"]).mean()
            actor_loss = actor_loss + bc_alpha * q_abs * bc_loss

        actor_info = {
            "actor/loss": actor_loss,
            "actor/entropy": -log_probs.mean(),
            "actor/mean_action": jnp.mean(actions),
        }
        return actor_loss, actor_info

    actor, info = actor.apply_gradient(actor_loss_fn)
    actor = l2normalize_network(actor)

    return actor, info


def update_critic_spl(
    key: PRNGKey,
    actor: Network,
    critic: Network,
    target_critic: Network,
    target_encoder: Network,
    target_lat_env_model: Network,
    temperature: Network,
    batch: Batch,
    use_cdq: bool,
    min_v: float,
    max_v: float,
    num_bins: int,
    gamma: float,
    n_step: int,
) -> Tuple[Network, Dict[str, float]]:
    """
    Update critic networks using categorical TD loss.

    Key difference: Critic takes [observation, z_state, action, z_next_state] with projection.

    Args:
        key: Random key
        actor: Actor network
        critic: Critic network
        target_critic: Target critic network
        target_encoder: Target encoder network
        target_lat_env_model: Target latent dynamics model
        temperature: Temperature network
        batch: Batch of transitions
        use_cdq: Whether to use Clipped Double Q-learning
        min_v: Minimum value for categorical distribution
        max_v: Maximum value for categorical distribution
        num_bins: Number of bins for categorical distribution
        gamma: Discount factor
        n_step: N-step returns

    Returns:
        new_critic: Updated critic
        info: Dictionary containing loss information
    """
    z_next_s = target_encoder(observations=batch["next_observation"])

    next_dist, _ = actor(
        observation=batch["next_observation"],
        z_state=z_next_s,
    )
    next_actions = next_dist.sample(seed=key)
    next_actor_log_probs = next_dist.log_prob(next_actions)
    next_actor_entropy = temperature() * next_actor_log_probs

    z_next_next_s = target_lat_env_model(z_state=z_next_s, action=next_actions)

    if use_cdq:
        next_qs, next_q_infos = target_critic(
            observation=batch["next_observation"],
            z_state=z_next_s,
            action=next_actions,
            z_next_state=z_next_next_s,
        )
        min_indices = next_qs.argmin(axis=0)
        next_q_log_probs = jax.vmap(
            lambda log_prob, idx: log_prob[idx], in_axes=(1, 0)
        )(next_q_infos["log_prob"], min_indices)
    else:
        _, next_q_info = target_critic(
            observation=batch["next_observation"],
            z_state=z_next_s,
            action=next_actions,
            z_next_state=z_next_next_s,
        )
        next_q_log_probs = next_q_info["log_prob"]

    z_s = target_encoder(observations=batch["observation"])
    z_next_s_pred = target_lat_env_model(z_state=z_s, action=batch["action"])

    def critic_loss_fn(
        critic_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        if use_cdq:
            pred_qs, pred_q_infos = critic.apply(
                variables={"params": critic_params},
                observation=batch["observation"],
                z_state=z_s,
                action=batch["action"],
                z_next_state=z_next_s_pred,
            )
            loss_1 = categorical_td_loss(
                pred_log_probs=pred_q_infos["log_prob"][0],
                target_log_probs=next_q_log_probs,
                reward=batch["reward"],
                done=batch["terminated"],
                actor_entropy=next_actor_entropy,
                gamma=gamma**n_step,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
            )
            loss_2 = categorical_td_loss(
                pred_log_probs=pred_q_infos["log_prob"][1],
                target_log_probs=next_q_log_probs,
                reward=batch["reward"],
                done=batch["terminated"],
                actor_entropy=next_actor_entropy,
                gamma=gamma**n_step,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
            )
            critic_loss = (loss_1 + loss_2).mean()
        else:
            pred_q, pred_q_info = critic.apply(
                variables={"params": critic_params},
                observation=batch["observation"],
                z_state=z_s,
                action=batch["action"],
                z_next_state=z_next_s_pred,
            )
            loss = categorical_td_loss(
                pred_log_probs=pred_q_info["log_prob"],
                target_log_probs=next_q_log_probs,
                reward=batch["reward"],
                done=batch["terminated"],
                actor_entropy=next_actor_entropy,
                gamma=gamma**n_step,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
            )
            critic_loss = loss.mean()

        critic_info = {
            "critic/loss": critic_loss,
            "critic/batch_rew_min": batch["reward"].min(),
            "critic/batch_rew_mean": batch["reward"].mean(),
            "critic/batch_rew_max": batch["reward"].max(),
        }
        return critic_loss, critic_info

    critic, info = critic.apply_gradient(critic_loss_fn)
    critic = l2normalize_network(critic)

    return critic, info
