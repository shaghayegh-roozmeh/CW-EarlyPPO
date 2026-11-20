'Clipped masked PPO with exact, independently counted interaction/update budgets.'
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.utils import explained_variance

from balanced_ppo import BalancedMaskablePPO, BalancedRolloutBuffer, clipped_actor_loss, weighted_mean
from ppo_wafer import EvaluateCallback, write_json


class UpdateEvaluateCallback(EvaluateCallback):
    """Validation every 5k real optimizer steps, never selected using TEST."""

    def _on_step(self):
        
        
        old_next = self.next_eval
        self.next_eval = math.inf
        result = super()._on_step()
        self.next_eval = old_next
        return result

    def after_optimizer_step(self):
        assert self.actual_gradient_steps == self.model.actual_optimizer_steps
        if self.actual_gradient_steps >= self.next_eval:
            self.perform()
            self.next_eval += self.args.eval_every

    def perform(self, final=False):
        result = super().perform(final=final)
        print(f'[actual updates] {self.actual_gradient_steps:,}', flush=True)
        return result

    def _on_training_end(self):
        assert self.num_timesteps == self.args.steps
        assert self.actual_gradient_steps == self.model.actual_optimizer_steps
        if self.args.target_kl is None:
            assert self.actual_gradient_steps == self.args.optimizer_steps
        super()._on_training_end()
        summary = json.loads((self.output/'training_summary.json').read_text())
        summary.update(optimizer_steps_requested=self.args.optimizer_steps,
            exact_update_budget_met=self.actual_gradient_steps == self.args.optimizer_steps,
            target_kl=self.args.target_kl, completed_rollouts=self.model.completed_rollouts,
            optimizer_updates_counted_by='Adam post-step hook, checked against explicit counter')
        write_json(self.output/'training_summary.json', summary)


class EqualBudgetPPO(BalancedMaskablePPO):
    """A hard budget limits genuine Adam steps, including the last partial epoch."""

    def __init__(self, *args, **kwargs):
        self.actual_optimizer_steps = 0
        self.rollout_optimizer_budget = 0
        self.completed_rollouts = 0
        self.budget_callback = None
        self.budget_output = None
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['budget_callback']

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_vf = (self.clip_range_vf(self._current_progress_remaining)
                   if self.clip_range_vf is not None else None)
        requested = int(self.rollout_optimizer_budget)
        assert requested > 0
        size = self.rollout_buffer.buffer_size * self.n_envs
        batches_per_epoch = math.ceil(size/self.batch_size)
        max_epochs = math.ceil(requested/batches_per_epoch)
        stats = {name: [] for name in ['approx_kl', 'unweighted_approx_kl',
            'policy_loss', 'value_loss', 'entropy_loss', 'clip_fraction', 'gradient_norm']}
        before = self.actual_optimizer_steps
        completed_epochs = 0
        stopped_by_kl = False
        for epoch in range(max_epochs):
            epoch_batches = 0
            for data in self.rollout_buffer.get(self.batch_size):
                if self.actual_optimizer_steps-before >= requested:
                    break
                actions = data.actions.long().flatten() if isinstance(self.action_space, spaces.Discrete) else data.actions
                values, log_prob, entropy = self.policy.evaluate_actions(
                    data.observations, actions, action_masks=data.action_masks)
                values = values.flatten()
                advantages = data.advantages
                if self.normalize_advantage:
                    advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
                log_ratio = log_prob-data.old_log_prob
                ratio = th.exp(log_ratio)
                weights = data.actor_weights
                policy_loss = clipped_actor_loss(advantages, ratio, weights, clip_range)
                value_prediction = values if clip_vf is None else data.old_values + th.clamp(
                    values-data.old_values, -clip_vf, clip_vf)
                
                value_loss = F.mse_loss(data.returns, value_prediction)
                entropy_loss = -weighted_mean(-log_prob if entropy is None else entropy, weights)
                loss = policy_loss+self.ent_coef*entropy_loss+self.vf_coef*value_loss
                with th.no_grad():
                    kl_terms = (th.exp(log_ratio)-1)-log_ratio
                    kl = weighted_mean(kl_terms, weights).item()
                    unweighted_kl = kl_terms.mean().item()
                    fraction = weighted_mean(((ratio-1).abs()>clip_range).float(), weights).item()
                if not (th.isfinite(loss) and np.isfinite(kl)):
                    raise FloatingPointError('Nonfinite PPO loss/KL; do not count a failed update')
                if self.target_kl is not None and kl > 1.5*self.target_kl:
                    stopped_by_kl = True
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                norm = th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                if not th.isfinite(norm):
                    raise FloatingPointError('Nonfinite gradient; do not count a failed update')
                self.policy.optimizer.step()
                self.actual_optimizer_steps += 1
                epoch_batches += 1
                for name, value in dict(approx_kl=kl, unweighted_approx_kl=unweighted_kl,
                    policy_loss=policy_loss.item(), value_loss=value_loss.item(),
                    entropy_loss=entropy_loss.item(), clip_fraction=fraction,
                    gradient_norm=norm.item()).items():
                    stats[name].append(value)
                if self.budget_callback is not None:
                    self.budget_callback.after_optimizer_step()
            
            self._n_updates += 1
            if epoch_batches == batches_per_epoch:
                completed_epochs += 1
            if stopped_by_kl or self.actual_optimizer_steps-before >= requested:
                break
        self.completed_rollouts += 1
        performed = self.actual_optimizer_steps-before
        if self.target_kl is None:
            assert performed == requested
        row = dict(rollout=self.completed_rollouts, env_steps=int(self.num_timesteps),
            rollout_transitions=size, requested_optimizer_steps=requested,
            actual_optimizer_steps=performed, cumulative_optimizer_steps=self.actual_optimizer_steps,
            full_epochs=completed_epochs, partial_epoch_batches=performed % batches_per_epoch,
            target_kl=self.target_kl, stopped_by_kl=stopped_by_kl,
            approx_kl_first=stats['approx_kl'][0] if performed else None,
            approx_kl_last=stats['approx_kl'][-1] if performed else None,
            approx_kl_max=max(stats['approx_kl']) if performed else None,
            fraction_updates_kl_above_previous_limit=float(np.mean(np.array(stats['approx_kl'])>.045)) if performed else None,
            **{name+'_mean':float(np.mean(values)) if values else None for name,values in stats.items()})
        if self.budget_output is not None:
            with (Path(self.budget_output)/'optimizer_diagnostics.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
        for name, values in stats.items():
            if values: self.logger.record('train/'+name, np.mean(values))
        self.logger.record('train/actual_optimizer_steps', self.actual_optimizer_steps)
        self.logger.record('train/explained_variance', explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()))

    def learn_exact(self, total_interactions, total_optimizer_steps, callback):
        if total_interactions % self.n_envs:
            raise ValueError('Exact interaction budget must be divisible by n_envs')
        if total_optimizer_steps != total_interactions:
            raise ValueError('This controlled runner is specifically one optimizer step per interaction')
        total_interactions, callback = self._setup_learn(total_interactions, callback,
            reset_num_timesteps=True, tb_log_name='EqualBudgetPPO', progress_bar=False)
        self.budget_callback = callback
        self.budget_output = str(callback.output)
        callback.on_training_start(locals(), globals())
        original_n_steps = self.n_steps
        try:
            while self.num_timesteps < total_interactions:
                remaining = total_interactions-self.num_timesteps
                steps = min(original_n_steps, remaining//self.n_envs)
                if self.rollout_buffer.buffer_size != steps:
                    self.rollout_buffer = BalancedRolloutBuffer(steps, self.observation_space,
                        self.action_space, device=self.device, gae_lambda=self.gae_lambda,
                        gamma=self.gamma, n_envs=self.n_envs, **self.rollout_buffer_kwargs)
                before = self.num_timesteps
                if not self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                             n_rollout_steps=steps, use_masking=True):
                    raise RuntimeError('Collection stopped before the requested budget')
                collected = self.num_timesteps-before
                assert collected == steps*self.n_envs
                self.rollout_optimizer_budget = collected
                self._update_current_progress_remaining(self.num_timesteps, total_interactions)
                self.train()
                self.logger.dump(step=self.num_timesteps)
            assert self.num_timesteps == total_interactions
            if self.target_kl is None:
                assert self.actual_optimizer_steps == total_optimizer_steps
            callback.on_training_end()
        finally:
            self.budget_callback = None
        return self
