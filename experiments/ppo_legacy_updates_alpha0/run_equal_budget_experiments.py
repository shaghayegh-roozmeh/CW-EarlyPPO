'Five predeclared Wafer experiments, each trained from scratch on TRAIN only.'
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from sklearn.model_selection import train_test_split
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from balanced_ppo import BalancedMaskablePPO, BalancedRolloutBuffer, inverse_frequency_weights, weighted_mean
from equal_budget_ppo import EqualBudgetPPO, UpdateEvaluateCallback
from ppo_wafer import PrefixCNN, PrefixWaferEnv, EvaluateCallback, evaluate, load_split, write_json, HASHES

EXPERIMENT = {'name': 'ppo_legacy_updates_alpha0', 'kind': 'legacy', 'alpha': 0.0, 'description': 'Current prefix CNN and natural class sampling, with the previous PPO training recipe: alpha=0, five epochs/rollout, target_kl=.03, 1,048,576 interactions.'}
ROOT = Path(__file__).resolve().parent


class CurrentOnlyWaferEnv(PrefixWaferEnv):
    """Only x_t is observed: no history, position, encoder, RNN or substitute."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(-np.inf, np.inf, (1,), np.float32)

    def observation(self):
        return self.X[self.index, self.t-1:self.t].copy()


class LegacyAuditMixin:
    """Instrument the unchanged legacy train() without modifying its updates."""
    def train(self):
        before = self.actual_optimizer_steps
        super().train()
        self.completed_rollouts += 1
        row = dict(rollout=self.completed_rollouts, env_steps=int(self.num_timesteps),
            rollout_transitions=int(self.rollout_buffer.buffer_size*self.n_envs),
            actual_optimizer_steps=self.actual_optimizer_steps-before,
            cumulative_optimizer_steps=self.actual_optimizer_steps,
            nominal_epochs=self.n_epochs, target_kl=self.target_kl,
            maximum_optimizer_steps=self.n_epochs*math.ceil(self.rollout_buffer.buffer_size*self.n_envs/self.batch_size))
        row['stopped_by_kl'] = row['actual_optimizer_steps'] < row['maximum_optimizer_steps']
        with (Path(self.audit_output)/'optimizer_diagnostics.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')


class LegacyMaskablePPO(LegacyAuditMixin, MaskablePPO):
    pass


class LegacyBalancedPPO(LegacyAuditMixin, BalancedMaskablePPO):
    pass


class LegacyCallback(EvaluateCallback):
    def _on_training_start(self):
        super()._on_training_start()
        def count_model(optimizer, args, kwargs):
            self.model.actual_optimizer_steps += 1
        self.model_counter_hook = self.model.policy.optimizer.register_step_post_hook(count_model)
        self.saved_100k = False

    def _on_step(self):
        result = super()._on_step()
        if self.num_timesteps == 100000 and not self.saved_100k:
            
            
            assert self.actual_gradient_steps == self.model.actual_optimizer_steps
            self.model.save(self.output/'budget100k_model')
            validation = evaluate(self.model, self.X, self.y, self.args.wait_lambda, self.args.wait_power)
            write_json(self.output/'budget100k_metadata.json', dict(
                env_steps=100000, actual_gradient_steps=self.actual_gradient_steps,
                latest_completed_training_rollout_env_steps=98304,
                snapshot='during collection, before the next legacy PPO update',
                checkpoint_selection='fixed 100000 interactions, not validation selected',
                validation=validation))
            self.saved_100k = True
            print('[fixed 100k snapshot]', json.dumps(validation), flush=True)
        return result

    def _on_training_end(self):
        assert self.saved_100k
        assert self.actual_gradient_steps == self.model.actual_optimizer_steps
        super()._on_training_end()
        self.model_counter_hook.remove()


class FixedPrefixClassifier(nn.Module):
    """Same prefix CNN and actor hidden layer; two class logits, no RL/critic."""
    def __init__(self):
        super().__init__()
        observation_space = spaces.Box(-np.inf, np.inf, (152,), np.float32)
        self.features_extractor = PrefixCNN(observation_space, features_dim=128)
        self.class_hidden = nn.Sequential(nn.Linear(128, 64), nn.Tanh())
        self.class_output = nn.Linear(64, 2)
        
        for module, gain in [(self.features_extractor, np.sqrt(2)),
                             (self.class_hidden, np.sqrt(2)), (self.class_output, .01)]:
            def initialize(layer, gain=gain):
                if isinstance(layer, (nn.Linear, nn.Conv1d)):
                    nn.init.orthogonal_(layer.weight, gain=gain)
                    if layer.bias is not None: nn.init.zeros_(layer.bias)
            module.apply(initialize)

    def forward(self, observation):
        return self.class_output(self.class_hidden(self.features_extractor(observation)))


def fixed_prefix_observations(X):
    observation = np.zeros_like(X, dtype=np.float32)
    observation[:, :2] = X[:, :2]
    return observation


def versions():
    return dict(python=platform.python_version(), torch=torch.__version__,
        sb3=__import__('stable_baselines3').__version__,
        sb3_contrib=__import__('sb3_contrib').__version__,
        gymnasium=__import__('gymnasium').__version__)


def preflight(X, y, train_ids, val_ids):
    reference = json.loads((ROOT/'PROVENANCE.json').read_text())['reference_split']
    assert train_ids.tolist() == reference['train_indices']
    assert val_ids.tolist() == reference['monitor_indices']
    assert np.bincount(y[train_ids], minlength=2).tolist() == [78, 722]
    assert np.bincount(y[val_ids], minlength=2).tolist() == [19, 181]
    env = CurrentOnlyWaferEnv(X, y, balanced=False)
    obs, _ = env.reset(options=dict(index=0, start_at=10))
    changed = X.copy(); changed[0, :9] = 123456.; changed[0, 10:] = -987654.
    other = CurrentOnlyWaferEnv(changed, y, balanced=False)
    altered, _ = other.reset(options=dict(index=0, start_at=10))
    assert obs.shape == (1,) and np.array_equal(obs, altered) and obs[0] == X[0, 9]
    next_obs, _, done, _, _ = env.step(0)
    assert not done and next_obs[0] == X[0, 10]
    env.reset(options=dict(index=0, start_at=152))
    assert env.action_masks().tolist() == [False, True, True]
    env = PrefixWaferEnv(X, y, balanced=False, random_start_fraction=0.)
    assert all(env.reset()[0].shape == (152,) and env.t == 1 for _ in range(100))
    assert np.count_nonzero(fixed_prefix_observations(X)[:, 2:]) == 0
    print('[preflight] exact split, current-only absence of past/future, WAIT mask, start-at-one and two-point input passed', flush=True)


def train_static(seed, directory, X, y, train_ids, val_ids, args):
    model = FixedPrefixClassifier()
    assert sum(p.numel() for p in model.parameters()) == 87042
    weights = torch.as_tensor(inverse_frequency_weights(y[train_ids], .5), dtype=torch.float32)
    features = torch.from_numpy(fixed_prefix_observations(X))
    labels = torch.as_tensor(y, dtype=torch.long)
    training_features, training_labels = features[train_ids], labels[train_ids]
    validation_features, validation_labels = features[val_ids], labels[val_ids]
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003, eps=1e-5)
    started = time.monotonic()
    best_loss, best_epoch, stale, updates = float('inf'), 0, 0, 0
    selected_path = directory/'selected_model.pt'
    for epoch in range(1, 257):
        model.train()
        permutation = torch.randperm(len(train_ids))
        training_loss = 0.
        for batch in permutation.split(256):
            logits = model(training_features[batch])
            per_item = nn.functional.cross_entropy(logits, training_labels[batch], reduction='none')
            loss = weighted_mean(per_item, weights[training_labels[batch]])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite supervised loss')
            optimizer.zero_grad(); loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), .5)
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite supervised gradient')
            optimizer.step(); updates += 1
            training_loss += loss.item()*len(batch)/len(train_ids)
        model.eval()
        with torch.no_grad():
            logits = model(validation_features)
            losses = nn.functional.cross_entropy(logits, validation_labels, reduction='none')
            validation_loss = weighted_mean(losses, weights[validation_labels]).item()
            predictions = logits.argmax(1).numpy()
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        improved = validation_loss < best_loss-1e-8
        if improved:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            torch.save(dict(state_dict=model.state_dict(), seed=seed, selected_epoch=epoch,
                optimizer_steps_at_selection=updates, selection='minimum weighted validation cross-entropy',
                validation_loss=validation_loss, input_prefix_length=2, model_parameters=87042), selected_path)
        else: stale += 1
        row = dict(epoch=epoch, actual_gradient_steps=updates, training_loss=training_loss,
            validation_loss=validation_loss, accuracy=float(accuracy_score(y[val_ids], predictions)),
            balanced_accuracy=float(balanced_accuracy_score(y[val_ids], predictions)),
            macro_f1=float(f1_score(y[val_ids], predictions, average='macro', zero_division=0)),
            avg_t=2., selected_epoch=best_epoch, seconds=time.monotonic()-started)
        with (directory/'monitor.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        if epoch % 10 == 0 or epoch == 1: print('[supervised validation]', json.dumps(row), flush=True)
        if stale >= 30: break
    selected = torch.load(selected_path, weights_only=False, map_location='cpu')
    assert selected['selected_epoch'] == best_epoch
    write_json(directory/'training_summary.json', dict(completed=True, seed=seed,
        algorithm='supervised fixed-prefix classifier; no PPO, WAIT, value function or environment interactions',
        env_steps=None, actual_gradient_steps=updates, completed_epochs=epoch,
        selected_epoch=best_epoch, optimizer_steps_at_selection=selected['optimizer_steps_at_selection'],
        checkpoint_selection='minimum weighted validation cross-entropy; patience 30, maximum 256 epochs',
        model_parameters=87042, wall_seconds=time.monotonic()-started))
    print('[completed supervised]', directory, flush=True)


def train_one(seed, output=None, threads=2):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    X, y = load_split(ROOT/'data', 'TRAIN')
    train_ids, val_ids = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=0)
    preflight(X, y, train_ids, val_ids)
    spec = EXPERIMENT
    directory = Path(output) if output else ROOT/'models'/f'seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'config.json').exists(): raise FileExistsError('Fresh training requires an empty output directory')
    legacy, static = spec['kind'] == 'legacy', spec['kind'] == 'static'
    alpha = float(spec.get('alpha', .5))
    weights = inverse_frequency_weights(y[train_ids], alpha)
    steps = 1048576 if legacy else (None if static else 100000)
    args = argparse.Namespace(experiment=spec['name'], mode='train', protocol='holdout', seed=seed,
        split_seed=0, train_order='split', steps=steps,
        optimizer_steps=None if legacy or static else 100000, natural_sampling=True,
        n_envs=None if static else 32, n_steps=None if static else 64, batch_size=256,
        epochs=256 if not legacy else 5, gae_lambda=.95, lr=.0003,
        ent_coef=.02, initial_wait_bias=0., init_checkpoint=None, exploration_mix=0.,
        wait_lambda=.0001, wait_power=1/3, correct_reward=1., incorrect_reward=-1.,
        actor_balance_alpha=None if static else alpha, supervised_class_balance_alpha=.5 if static else None,
        balance_value_loss=False, reward_balance_alpha=0., reward_class_weights=[1., 1.],
        random_start_fraction=0. if spec['kind']=='start1' else .5, threads=threads,
        sampler='standard', max_validation_mean_t=0., save_reference_at=0,
        eval_every=32768 if legacy else 5000,
        eval_every_unit='environment_interactions' if legacy else 'actual_optimizer_steps',
        target_kl=.03 if legacy else None,
        data_dir=str(ROOT/'data'), output=str(directory), exact_budget=not legacy and not static,
        observation_mode='current_scalar_only' if spec['kind']=='current_only' else 'right_zero_padded_observed_prefix',
        input_prefix_length=2 if static else None, algorithm=spec['description'])
    if static:
        args.static_max_epochs=256; args.static_early_stopping_patience=30
        args.static_checkpoint_selection='minimum weighted validation cross-entropy'
        args.random_start_fraction=None; args.ent_coef=None; args.gae_lambda=None
    write_json(directory/'config.json', vars(args))
    write_json(directory/'split_manifest.json', dict(dataset='original UCR2015 Wafer',
        train_hash=HASHES['TRAIN'], train_indices=train_ids.tolist(), monitor_indices=val_ids.tolist(),
        monitor='independent_validation', test_used_for_selection=False, observation=args.observation_mode,
        actor_class_weights=None if static else weights.tolist(),
        supervised_class_weights=weights.tolist() if static else None,
        actor_balance_alpha=args.actor_balance_alpha, sampler='standard',
        reward=dict(correct=1., incorrect=-1., wait_lambda=.0001, wait_power=1/3, class_weights=[1., 1.]),
        classification_pretraining=False, auxiliary_classification_loss=False,
        initialization='fresh random weights; no historical checkpoints reused', versions=versions()))
    if static:
        train_static(seed, directory, X, y, train_ids, val_ids, args)
        return
    env_class = CurrentOnlyWaferEnv if spec['kind']=='current_only' else PrefixWaferEnv
    def make_env(i):
        return lambda: env_class(X[train_ids], y[train_ids], seed=seed+i, wait_lambda=.0001,
            wait_power=1/3, balanced=False, random_start_fraction=args.random_start_fraction,
            sampler='standard')
    vec = DummyVecEnv([make_env(i) for i in range(32)])
    policy_kwargs = dict(net_arch=dict(pi=[64], vf=[64]), activation_fn=nn.Tanh)
    if spec['kind']=='current_only':
        policy_kwargs['features_extractor_class'] = FlattenExtractor
    else:
        policy_kwargs.update(features_extractor_class=PrefixCNN, features_extractor_kwargs=dict(features_dim=128))
    balanced = not legacy or alpha != 0.
    model_class = (LegacyBalancedPPO if alpha else LegacyMaskablePPO) if legacy else EqualBudgetPPO
    extra = dict(rollout_buffer_class=BalancedRolloutBuffer,
        rollout_buffer_kwargs=dict(class_weights=weights)) if balanced else {}
    model = model_class('MlpPolicy', vec, learning_rate=.0003, n_steps=64, batch_size=256,
        n_epochs=args.epochs, gamma=1., gae_lambda=.95, clip_range=.2, ent_coef=.02,
        vf_coef=.5, max_grad_norm=.5, target_kl=args.target_kl, seed=seed,
        device='cpu', verbose=0, policy_kwargs=policy_kwargs, **extra)
    parameters = sum(p.numel() for p in model.policy.parameters())
    assert parameters == (516 if spec['kind']=='current_only' else 95428)
    assert sum(p.numel() for p in model.policy.features_extractor.parameters()) == (0 if spec['kind']=='current_only' else 78656)
    if legacy:
        model.actual_optimizer_steps=0; model.completed_rollouts=0; model.audit_output=str(directory)
    callback_class = LegacyCallback if legacy else UpdateEvaluateCallback
    callback = callback_class(X[val_ids], y[val_ids], directory, args)
    print('[start]', spec['name'], 'seed', seed, 'interactions', steps, 'parameters', parameters,
          'alpha', alpha, 'epochs', args.epochs, 'target_kl', args.target_kl, flush=True)
    started = time.monotonic()
    try:
        if legacy: model.learn(total_timesteps=steps, callback=callback)
        else: model.learn_exact(100000, 100000, callback)
    finally:
        vec.close()
    summary = json.loads((directory/'training_summary.json').read_text())
    adam_steps = {int(s['step']) for s in model.policy.optimizer.state.values() if 'step' in s}
    assert adam_steps == {summary['actual_gradient_steps']}
    assert summary['env_steps'] == steps
    summary.update(seed=seed, wall_seconds=time.monotonic()-started, target_kl=args.target_kl,
        completed_rollouts=model.completed_rollouts, epochs_per_full_rollout=args.epochs,
        actual_adam_state_steps=list(adam_steps), fresh_random_initialization=True,
        fixed_final_checkpoint=True, observation_mode=args.observation_mode)
    write_json(directory/'training_summary.json', summary)
    for name in ['best_model.zip', 'best_monitor.json', 'early_best_model.zip', 'early_best_monitor.json']:
        path=directory/name
        if path.exists(): path.unlink()
    print('[completed]', spec['name'], seed, json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output')
    parser.add_argument('--threads', type=int, default=2)
    a = parser.parse_args()
    train_one(a.seed, a.output, a.threads)
