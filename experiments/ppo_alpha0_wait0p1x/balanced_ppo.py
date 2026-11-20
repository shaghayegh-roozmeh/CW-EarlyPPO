'Class-weighted PPO actor ablation based on sb3-contrib 2.7.0.'
from typing import NamedTuple

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from stable_baselines3.common.utils import explained_variance


class BalancedSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor
    actor_weights: th.Tensor


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def clipped_actor_loss(advantages, ratio, weights, clip_range):
    unclipped = advantages * ratio
    clipped = advantages * th.clamp(ratio, 1-clip_range, 1+clip_range)
    return -weighted_mean(th.minimum(unclipped, clipped), weights)


class BalancedRolloutBuffer(MaskableRolloutBuffer):
    def __init__(self, *args, class_weights=(1., 1.), per_transition_weights=False, **kwargs):
        self.class_weights = np.asarray(class_weights, dtype=np.float32)
        self.per_transition_weights = bool(per_transition_weights)
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.class_labels = np.full((self.buffer_size, self.n_envs, 1), -1, np.int64)
        self.transition_weights = np.full((self.buffer_size, self.n_envs, 1), np.nan, np.float32)

    def add(self, *args, **kwargs):
        if not np.isin(self.class_labels[self.pos], [0, 1]).all():
            raise RuntimeError('Training class metadata must be set before buffer.add')
        if self.per_transition_weights and not (self.transition_weights[self.pos]>0).all():
            raise RuntimeError('Actual episode-distribution weights must be set before buffer.add')
        return super().add(*args, **kwargs)

    def get(self, batch_size=None):
        if not self.generator_ready:
            self.class_labels = self.swap_and_flatten(self.class_labels)
            self.transition_weights = self.swap_and_flatten(self.transition_weights)
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env=None):
        base = super()._get_samples(batch_inds, env)
        labels = self.class_labels[batch_inds].reshape(-1)
        weights = self.to_torch(self.transition_weights[batch_inds].flatten()
            if self.per_transition_weights else self.class_weights[labels])
        return BalancedSamples(*base, weights)


class BalancedMaskablePPO(MaskablePPO):
    """Same clipped PPO update, with class weights in actor and entropy losses."""
    def __init__(self, *args, balance_value_loss=False, **kwargs):
        self.balance_value_loss = bool(balance_value_loss)
        super().__init__(*args, **kwargs)

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for data in self.rollout_buffer.get(self.batch_size):
                actions = data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(
                    data.observations, actions, action_masks=data.action_masks)
                values = values.flatten()
                advantages = data.advantages
                if self.normalize_advantage:
                    
                    advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
                ratio = th.exp(log_prob-data.old_log_prob)
                weights = data.actor_weights
                policy_loss = clipped_actor_loss(advantages, ratio, weights, clip_range)
                pg_losses.append(policy_loss.item())
                clip_fractions.append(weighted_mean(
                    ((ratio-1).abs()>clip_range).float(), weights).item())
                values_pred = values if self.clip_range_vf is None else data.old_values + th.clamp(
                    values-data.old_values, -clip_range_vf, clip_range_vf)
                
                value_loss = (weighted_mean((data.returns-values_pred).square(), weights)
                              if self.balance_value_loss else F.mse_loss(data.returns, values_pred))
                value_losses.append(value_loss.item())
                entropy_loss = -weighted_mean(-log_prob if entropy is None else entropy, weights)
                entropy_losses.append(entropy_loss.item())
                loss = policy_loss+self.ent_coef*entropy_loss+self.vf_coef*value_loss
                with th.no_grad():
                    log_ratio = log_prob-data.old_log_prob
                    approx_kl = weighted_mean((th.exp(log_ratio)-1)-log_ratio, weights).item()
                    approx_kl_divs.append(approx_kl)
                if self.target_kl is not None and approx_kl > 1.5*self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            if not continue_training:
                break
        self._n_updates += self.n_epochs
        variance = explained_variance(self.rollout_buffer.values.flatten(),
                                      self.rollout_buffer.returns.flatten())
        for name, value in dict(entropy_loss=np.mean(entropy_losses),
                policy_gradient_loss=np.mean(pg_losses), value_loss=np.mean(value_losses),
                approx_kl=np.mean(approx_kl_divs), clip_fraction=np.mean(clip_fractions),
                loss=loss.item(), explained_variance=variance,
                clip_range=clip_range).items():
            self.logger.record('train/'+name, value)
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')


def inverse_frequency_weights(labels, alpha):
    counts = np.bincount(np.asarray(labels, np.int64), minlength=2)
    if (counts == 0).any():
        raise ValueError('Both classes are required')
    frequency = counts / counts.sum()
    weights = (1/(2*frequency)) ** float(alpha)
    
    return weights / np.dot(frequency, weights)
