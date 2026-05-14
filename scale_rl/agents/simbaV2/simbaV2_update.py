from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp

from scale_rl.agents.jax_utils.network import Network, PRNGKey
from scale_rl.agents.jax_utils.tree_utils import tree_map_until_match
from scale_rl.buffers import Batch

EPS = 1e-8


def l2normalize(
    x: jnp.ndarray,
    axis: int,
) -> jnp.ndarray:
    l2norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    x = x / jnp.maximum(l2norm, EPS)

    return x


def l2normalize_layer(tree):
    """
    apply l2-normalization to the all leaf nodes
    """
    if len(tree["kernel"].shape) == 2:
        axis = 0
    elif len(tree["kernel"].shape) == 3:
        axis = 1
    else:
        raise ValueError
    return jax.tree.map(f=lambda x: l2normalize(x, axis=axis), tree=tree)


def l2normalize_network(
    network: Network,
    regex: str = "hyper_dense",
) -> Network:
    params = network.params
    new_params = tree_map_until_match(
        f=lambda x: l2normalize_layer(x), tree=params, target_re=regex, keep_values=True
    )
    network = network.replace(params=new_params)
    return network


def update_actor(
    key: PRNGKey,
    actor: Network,
    critic: Network,
    temperature: Network,
    batch: Batch,
    use_cdq: bool,
    bc_alpha: float,
) -> Tuple[Network, Dict[str, float]]:
    def actor_loss_fn(
        actor_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        dist, _ = actor.apply(
            variables={"params": actor_params},
            observations=batch["observation"],
        )
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)

        if use_cdq:
            # qs: (2, n)
            qs, q_infos = critic(observations=batch["observation"], actions=actions)
            q = jnp.minimum(qs[0], qs[1])
        else:
            q, _ = critic(observations=batch["observation"], actions=actions)

        actor_loss = (log_probs * temperature() - q).mean()

        if bc_alpha > 0:
            # https://arxiv.org/abs/2306.02451
            q_abs = jax.lax.stop_gradient(jnp.abs(q).mean())
            bc_loss = ((actions - batch["action"]) ** 2).mean()
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


def categorical_td_loss(
    pred_log_probs: jnp.ndarray,  # (n, num_bins)
    target_log_probs: jnp.ndarray,  # (n, num_bins)
    reward: jnp.ndarray,  # (n,)
    done: jnp.ndarray,  # (n,)
    actor_entropy: jnp.ndarray,  # (n,)
    gamma: float,
    num_bins: int,
    min_v: float,
    max_v: float,
) -> jnp.ndarray:
    reward = reward.reshape(-1, 1)
    done = done.reshape(-1, 1)
    actor_entropy = actor_entropy.reshape(-1, 1)

    # compute target value buckets
    # target_bin_values: (n, num_bins)
    bin_values = jnp.linspace(start=min_v, stop=max_v, num=num_bins).reshape(1, -1)
    target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
    target_bin_values = jnp.clip(target_bin_values, min_v, max_v)  # (B, num_bins)

    # update indices
    b = (target_bin_values - min_v) / ((max_v - min_v) / (num_bins - 1))
    l = jnp.floor(b)
    l_mask = jax.nn.one_hot(l.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))
    u = jnp.ceil(b)
    u_mask = jax.nn.one_hot(u.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))

    # target label
    _target_probs = jnp.exp(target_log_probs)
    m_l = (_target_probs * (u + (l == u).astype(jnp.float32) - b)).reshape(
        -1, num_bins, 1
    )
    m_u = (_target_probs * (b - l)).reshape((-1, num_bins, 1))
    target_probs = jax.lax.stop_gradient(jnp.sum(m_l * l_mask + m_u * u_mask, axis=1))

    # cross entropy loss
    loss = -jnp.mean(jnp.sum(target_probs * pred_log_probs, axis=1))

    return loss


def update_critic(
    key: PRNGKey,
    actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    batch: Batch,
    use_cdq: bool,
    min_v: float,
    max_v: float,
    num_bins: int,
    gamma: float,
    n_step: int,
) -> Tuple[Network, Dict[str, float]]:
    # compute the target q-value
    next_dist, _ = actor(observations=batch["next_observation"])
    next_actions = next_dist.sample(seed=key)
    next_actor_log_probs = next_dist.log_prob(next_actions)
    next_actor_entropy = temperature() * next_actor_log_probs

    if use_cdq:
        # next_qs: (2, n)
        # next_q_infos['log_prob]: (2, n, num_bins)
        # next_q_log_probs: (n, num_bins)
        next_qs, next_q_infos = target_critic(
            observations=batch["next_observation"], actions=next_actions
        )
        min_indices = next_qs.argmin(axis=0)
        next_q_log_probs = jax.vmap(
            lambda log_prob, idx: log_prob[idx], in_axes=(1, 0)
        )(next_q_infos["log_prob"], min_indices)
    else:
        next_q, next_q_info = target_critic(
            observations=batch["next_observation"],
            actions=next_actions,
        )
        next_q_log_probs = next_q_info["log_prob"]

    def critic_loss_fn(
        critic_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        if use_cdq:
            # compute predicted q-value
            pred_qs, pred_q_infos = critic.apply(
                variables={"params": critic_params},
                observations=batch["observation"],
                actions=batch["action"],
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
                observations=batch["observation"],
                actions=batch["action"],
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


def update_target_network(
    network: Network,
    target_network: Network,
    target_tau: bool,
) -> Tuple[Network, Dict[str, float]]:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * target_tau + tp * (1 - target_tau),
        network.params,
        target_network.params,
    )
    target_network = target_network.replace(params=new_target_params)

    info = {}

    return target_network, info


def update_temperature(
    temperature: Network, entropy: float, target_entropy: float
) -> Tuple[Network, Dict[str, float]]:
    def temperature_loss_fn(
        temperature_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        temperature_value = temperature.apply({"params": temperature_params})
        temperature_loss = temperature_value * (entropy - target_entropy).mean()
        temperature_info = {
            "temperature/value": temperature_value,
            "temperature/loss": temperature_loss,
        }

        return temperature_loss, temperature_info

    temperature, info = temperature.apply_gradient(temperature_loss_fn)

    return temperature, info
