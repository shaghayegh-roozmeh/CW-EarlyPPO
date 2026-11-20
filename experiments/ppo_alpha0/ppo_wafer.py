'On-policy PPO for the Martinez early-classification formulation on UCR2015.'
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from balanced_ppo import BalancedMaskablePPO, BalancedRolloutBuffer, inverse_frequency_weights

WAIT, CLASS0, CLASS1 = 0, 1, 2
HASHES = {
    'TRAIN': '31bba0c41bd068e5b9f3fc04ce7856dc23d93758d2e09cc53ca4a32d3b7a0f05',
    'TEST': 'bfce80cbcb14d2623c64e70c81ef8a83204c79a5fa739cd247c78b5b34610605',
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_split(directory, split):
    path = Path(directory) / f'wafer_{split}'
    if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[split]:
        raise ValueError(f'{path} does not match the verified original UCR2015 bytes')
    a = np.loadtxt(path, delimiter=',', dtype=np.float32)
    assert a.shape == ((1000 if split == 'TRAIN' else 6164), 153)
    assert np.isfinite(a).all() and set(np.unique(a[:, 0])) == {-1.0, 1.0}
    return a[:, 1:].copy(), (a[:, 0] == 1).astype(np.int64)


class PrefixWaferEnv(gym.Env):
    'Self-contained observed prefix, right-padded with zeros to length T.'
    metadata = {'render_modes': []}

    def __init__(self, X, y, seed=0, wait_lambda=0.001, wait_power=1/3,
                 balanced=True, random_start_fraction=0.5,
                 correct_reward=1., incorrect_reward=-1., reward_class_weights=(1., 1.),
                 sampler='standard'):
        super().__init__()
        self.X, self.y = np.asarray(X, np.float32), np.asarray(y, np.int64)
        self.T = self.X.shape[1]
        self.wait_lambda, self.wait_power = float(wait_lambda), float(wait_power)
        self.balanced, self.random_start_fraction = balanced, random_start_fraction
        self.correct_reward, self.incorrect_reward = float(correct_reward), float(incorrect_reward)
        self.reward_class_weights = np.asarray(reward_class_weights, np.float32)
        self.sampler, self.training_progress = sampler, 0.
        self.natural_probabilities = np.bincount(self.y,minlength=2)/len(self.y)
        self.by_class = [np.flatnonzero(self.y == c) for c in (0, 1)]
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.T,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.index, self.t, self.done = 0, 1, True

    def set_training_progress(self, fraction):
        self.training_progress = float(np.clip(fraction,0,1))

    def sampling_probabilities(self):
        if self.sampler=='class_anneal':
            minority=int(np.argmin(self.natural_probabilities))
            fraction=min(self.training_progress/.5,1.)
            p_minority=(1-fraction)*.5+fraction*self.natural_probabilities[minority]
            return np.array([p_minority,1-p_minority]) if minority==0 else np.array([1-p_minority,p_minority])
        return np.array([.5,.5]) if self.balanced else self.natural_probabilities.copy()

    def curriculum_lower_bound(self):
        if self.sampler!='prefix_curriculum' or self.training_progress>=.6:
            return None
        initial=int(np.ceil(.75*self.T))
        fraction=np.clip((self.training_progress-.2)/.4,0,1)
        return int(np.ceil(1+(initial-1)*(1-fraction)))

    def observation(self):
        obs = np.zeros(self.T, np.float32)
        obs[:self.t] = self.X[self.index, :self.t]
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        probabilities=self.sampling_probabilities()
        if 'index' in options:
            self.index = int(options['index'])
        elif self.sampler=='class_anneal':
            label=0 if self.rng.random()<probabilities[0] else 1
            self.index=int(self.rng.choice(self.by_class[label]))
        elif self.balanced:
            self.index = int(self.rng.choice(self.by_class[int(self.rng.integers(2))]))
        else:
            self.index = int(self.rng.integers(len(self.y)))
        if 'start_at' in options:
            self.t = int(options['start_at'])
        else:
            lower=self.curriculum_lower_bound()
            self.t = (int(self.rng.integers(lower,self.T+1)) if lower is not None else
                (int(self.rng.integers(1, self.T + 1))
                 if self.rng.random() < self.random_start_fraction else 1))
        assert 1 <= self.t <= self.T
        self.done = False
        self.episode_class_probability=float(probabilities[self.y[self.index]])
        self.episode_start_t=self.t
        return self.observation(), {}

    def action_masks(self):
        return np.array([self.t < self.T, True, True], dtype=bool)

    def step(self, action):
        action = int(action)
        if self.done:
            raise RuntimeError('Reset required after an episode')
        if action == WAIT:
            if self.t >= self.T:
                raise ValueError('WAIT is masked at the final measurement')
            reward = -self.wait_lambda * self.t ** self.wait_power * self.reward_class_weights[self.y[self.index]]
            info = {'label': int(self.y[self.index]), 'series_index': self.index,
                    'observation_time': self.t,
                    'episode_class_probability': self.episode_class_probability,
                    'episode_start_t': self.episode_start_t}
            self.t += 1
            return self.observation(), float(reward), False, False, info
        if action not in (CLASS0, CLASS1):
            raise ValueError(action)
        pred = action - 1
        correct = pred == self.y[self.index]
        reward = (self.correct_reward if correct else self.incorrect_reward) * self.reward_class_weights[self.y[self.index]]
        self.done = True
        return self.observation(), reward, True, False, {
            'prediction': pred, 'label': int(self.y[self.index]),
            'decision_time': self.t, 'correct': bool(correct),
            'series_index': self.index, 'observation_time': self.t,
            'episode_class_probability': self.episode_class_probability,
            'episode_start_t': self.episode_start_t,
        }


class PrefixCNN(BaseFeaturesExtractor):
    """Small 1-D CNN over the observed prefix, with no normalization layers."""
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(8, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            size = self.conv(torch.zeros(1, 1, observation_space.shape[0])).shape[1]
        self.linear = nn.Sequential(nn.Linear(size, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.conv(observations.unsqueeze(1)))


class ExploringPolicy(MaskableActorCriticPolicy):
    'A declared epsilon-uniform policy family with exact on-policy log-probs.'
    def __init__(self, *args, exploration_mix=0.1, **kwargs):
        self.exploration_mix = float(exploration_mix)
        super().__init__(*args, **kwargs)

    def _get_action_dist_from_latent(self, latent_pi):
        logits = self.action_net(latent_pi)
        probs = logits.softmax(-1)
        mixed = (1 - self.exploration_mix) * probs + self.exploration_mix / probs.shape[-1]
        return self.action_dist.proba_distribution(action_logits=mixed.log())


@torch.no_grad()
def evaluate(model, X, y, wait_lambda=0.001, wait_power=1/3,
             stochastic=False, seed=0, return_predictions=False,
             correct_reward=1., incorrect_reward=-1., reward_class_weights=(1., 1.)):
    """Every evaluation starts from t=1, with natural data frequencies."""
    old_mode = model.policy.training
    model.policy.set_training_mode(False)
    n, T = X.shape
    pred = np.full(n, -1, np.int64)
    stops = np.full(n, T, np.int64)
    first_probs = np.zeros((n, 3), np.float32)
    action_counts = np.zeros(3, np.int64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for t in range(1, T + 1):
            active = np.flatnonzero(pred < 0)
            if not len(active):
                break
            obs = np.zeros((len(active), T), np.float32)
            obs[:, :t] = X[active, :t]
            masks = np.ones((len(active), 3), bool)
            masks[:, WAIT] = t < T
            tensors = torch.as_tensor(obs, device=model.device)
            dist = model.policy.get_distribution(tensors, action_masks=masks)
            probs = dist.distribution.probs
            actions = (torch.multinomial(probs, 1).squeeze(-1) if stochastic
                       else probs.argmax(-1)).cpu().numpy()
            if t == 1:
                first_probs[active] = probs.cpu().numpy()
            action_counts += np.bincount(actions, minlength=3)
            chosen = actions != WAIT
            pred[active[chosen]] = actions[chosen] - 1
            stops[active[chosen]] = t
    model.policy.set_training_mode(old_mode)
    assert (pred >= 0).all()
    cm = confusion_matrix(y, pred, labels=[0, 1])
    costs = np.r_[0., np.cumsum(wait_lambda * np.arange(1, T) ** wait_power)]
    ret = (np.where(pred == y, correct_reward, incorrect_reward)-costs[stops-1]) * np.asarray(reward_class_weights)[y]
    result = dict(
        n=int(n), evaluation='sample' if stochastic else 'greedy',
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average='macro', zero_division=0)),
        confusion_matrix=cm.tolist(),
        recall_by_class=[float(cm[c, c] / cm[c].sum()) for c in (0, 1)],
        avg_t=float(stops.mean()), earliness=float(stops.mean() / T),
        first_stop_fraction=float((stops == 1).mean()), mean_return=float(ret.mean()),
        avg_t_by_class=[float(stops[y == c].mean()) for c in (0, 1)],
        action_counts=action_counts.tolist(),
        first_probs_by_class=[first_probs[y == c].mean(0).tolist() for c in (0, 1)],
    )
    if return_predictions:
        return result, dict(labels=y, predictions=pred, decision_times=stops)
    return result


class EvaluateCallback(BaseCallback):
    def __init__(self, X, y, output, args):
        super().__init__()
        self.X, self.y, self.output, self.args = X, y, Path(output), args
        self.best_key = (-1., -float('inf'))
        self.next_eval = args.eval_every
        self.started = time.monotonic()
        self.best_early_key=(-1.,-float('inf'))
        self.saved_reference=False
        self.actual_gradient_steps=0

    def _on_training_start(self):
        def count_update(optimizer,args,kwargs):
            self.actual_gradient_steps+=1
        self.optimizer_hook=self.model.policy.optimizer.register_step_post_hook(count_update)

    def _on_rollout_start(self):
        self.rollout_labels, self.rollout_actions = [], []
        self.rollout_probabilities,self.rollout_starts=[],[]
        if self.args.sampler!='standard':
            self.training_env.env_method('set_training_progress',self.num_timesteps/self.args.steps)
        if self.args.save_reference_at and self.num_timesteps==self.args.save_reference_at and not self.saved_reference:
            self.model.save(self.output/'reference_1m_model')
            self.saved_reference=True

    def _on_rollout_end(self):
        labels = np.asarray(self.rollout_labels)
        actions = np.asarray(self.rollout_actions)
        buffer = self.model.rollout_buffer
        weights = (buffer.class_weights[labels] if isinstance(buffer, BalancedRolloutBuffer)
                   else np.ones_like(labels, dtype=np.float32))
        if isinstance(buffer,BalancedRolloutBuffer) and buffer.per_transition_weights:
            weights=buffer.transition_weights[:,:,0]
        counts = np.bincount(labels.flatten(), minlength=2)
        row = dict(env_steps=int(self.num_timesteps), transition_class_counts=counts.tolist(),
            sampler=self.args.sampler,actual_gradient_steps=self.actual_gradient_steps,
            mean_episode_start=float(np.mean(self.rollout_starts)),
            mean_episode_class_probability=float(np.mean(self.rollout_probabilities)),
            weighted_class_mass=[float(weights[labels==c].sum()/weights.sum()) for c in (0,1)],
            action_counts_by_class=[np.bincount(actions[labels==c],minlength=3).tolist() for c in (0,1)],
            mean_advantage_by_class=[float(buffer.advantages[labels==c].mean()) if counts[c] else None for c in (0,1)],
            mean_return_by_class=[float(buffer.returns[labels==c].mean()) if counts[c] else None for c in (0,1)])
        with (self.output/'rollout_diagnostics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')

    def perform(self, final=False):
        a = self.args
        result = evaluate(self.model, self.X, self.y, a.wait_lambda, a.wait_power,
            correct_reward=a.correct_reward, incorrect_reward=a.incorrect_reward,
            reward_class_weights=a.reward_class_weights)
        result.update(env_steps=int(self.num_timesteps), ppo_epochs=int(self.model._n_updates),
                      actual_gradient_steps=self.actual_gradient_steps,
                      seconds=time.monotonic() - self.started, final=final)
        with (self.output / 'monitor.jsonl').open('a') as f:
            f.write(json.dumps(result) + '\n')
        key = (result['accuracy'], result['mean_return'])
        if key > self.best_key:
            self.best_key = key
            self.model.save(self.output / 'best_model')
            write_json(self.output / 'best_monitor.json', result)
        if self.args.max_validation_mean_t and result['avg_t']<=self.args.max_validation_mean_t and key>self.best_early_key:
            self.best_early_key=key
            self.model.save(self.output/'early_best_model')
            write_json(self.output/'early_best_monitor.json',result)
        self.model.save(self.output / 'last_model')
        print(f"[eval] step={self.num_timesteps:,} acc={result['accuracy']:.5f} "
              f"BA={result['balanced_accuracy']:.5f} t={result['avg_t']:.3f} "
              f"return={result['mean_return']:.4f} elapsed={result['seconds']:.1f}s "
              f"best_acc={self.best_key[0]:.5f}", flush=True)
        return result

    def _on_step(self):
        self.rollout_labels.append([info['label'] for info in self.locals['infos']])
        self.rollout_actions.append(np.asarray(self.locals['actions']).reshape(-1).tolist())
        self.rollout_probabilities.append([info['episode_class_probability'] for info in self.locals['infos']])
        self.rollout_starts.append([info['episode_start_t'] for info in self.locals['infos']])
        buffer = self.locals.get('rollout_buffer')
        if isinstance(buffer, BalancedRolloutBuffer):
            labels = np.array([info['label'] for info in self.locals['infos']], dtype=np.int64)
            buffer.class_labels[buffer.pos, :, 0] = labels
            if buffer.per_transition_weights:
                probability=np.asarray(self.rollout_probabilities[-1],dtype=np.float64)
                alpha=self.args.actor_balance_alpha
                raw=(1/(2*probability))**alpha
                normalizer=probability*raw+(1-probability)*(1/(2*(1-probability)))**alpha
                buffer.transition_weights[buffer.pos,:,0]=(raw/normalizer).astype(np.float32)
        if self.num_timesteps >= self.next_eval:
            self.perform()
            self.next_eval += self.args.eval_every
        return True

    def _on_training_end(self):
        self.perform(final=True)
        write_json(self.output/'training_summary.json',dict(completed=True,
            env_steps=int(self.num_timesteps),actual_gradient_steps=self.actual_gradient_steps,
            model_parameters=sum(p.numel() for p in self.model.policy.parameters())))
        self.optimizer_hook.remove()


def check_environment(X, y):
    """Meaningful task checks before a potentially long training run."""
    env = PrefixWaferEnv(X, y, seed=0)
    ix = int(np.flatnonzero(y == 0)[0])
    obs, _ = env.reset(options=dict(index=ix, start_at=4))
    altered = X.copy(); altered[:, 4:] = 1e6
    other = PrefixWaferEnv(altered, y)
    obs2, _ = other.reset(options=dict(index=ix, start_at=4))
    assert np.array_equal(obs, obs2), 'Unobserved suffix leaked into policy inputs'
    assert np.array_equal(obs[:4], X[ix, :4]) and np.count_nonzero(obs[4:]) == 0
    after, reward, done, trunc, _ = env.step(WAIT)
    assert np.isclose(reward, -.001 * 4 ** (1/3)) and not done and not trunc
    assert np.array_equal(after[:5], X[ix, :5]) and np.count_nonzero(after[5:]) == 0
    for label in (0, 1):
        ix = int(np.flatnonzero(y == label)[0])
        for action in (CLASS0, CLASS1):
            env.reset(options=dict(index=ix, start_at=16))
            _, r, done, _, info = env.step(action)
            assert r == (1. if action - 1 == label else -1.) and done
            assert info['decision_time'] == 16
    env.reset(options=dict(index=ix, start_at=X.shape[1]))
    assert env.action_masks().tolist() == [False, True, True]
    print('[checks] prefix causality, action/reward mapping, WAIT cost and final mask passed', flush=True)


def train(args):
    if args.balance_value_loss and not args.actor_balance_alpha:
        raise ValueError('--balance-value-loss requires --actor-balance-alpha > 0')
    if args.correct_reward <= args.incorrect_reward or args.wait_lambda < 0:
        raise ValueError('Correct decisions must pay more than incorrect ones; WAIT cost must be nonnegative')
    if args.sampler!='standard' and args.reward_balance_alpha:
        raise ValueError('These controlled sampler experiments keep rewards unweighted')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    X, y = load_split(args.data_dir, 'TRAIN')
    check_environment(X, y)
    indices = np.arange(len(y))
    if args.protocol == 'holdout':
        train_ids, monitor_ids = train_test_split(indices, test_size=.2,
            stratify=y, random_state=args.split_seed)
        monitor_name = 'independent_validation'
    else:
        train_ids = monitor_ids = indices
        monitor_name = 'in_sample_train_monitor'
    if args.train_order == 'sorted':
        train_ids = np.sort(train_ids)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.json').exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    
    
    sampling_labels = y[train_ids] if args.natural_sampling else np.array([0,1])
    actor_weights = inverse_frequency_weights(sampling_labels, args.actor_balance_alpha)
    args.reward_class_weights = inverse_frequency_weights(sampling_labels, args.reward_balance_alpha).tolist()
    write_json(output / 'config.json', vars(args))
    write_json(output / 'split_manifest.json', dict(
        dataset='original UCR2015 Wafer', train_hash=HASHES['TRAIN'],
        train_indices=train_ids.tolist(), monitor_indices=monitor_ids.tolist(),
        monitor=monitor_name, test_used_for_selection=False,
        observation='right-zero-padded real prefix; no clock, label, or future values',
        algorithm='SB3-contrib MaskablePPO, real on-policy rollouts',
        classification_pretraining=False, auxiliary_classification_loss=False,
        final_action_set='CLASS0, CLASS1; WAIT invalid at T',
        reward=dict(correct=args.correct_reward, incorrect=args.incorrect_reward,
                    wait_lambda=args.wait_lambda, wait_power=args.wait_power,
                    class_weights=args.reward_class_weights),
        actor_class_weights=actor_weights.tolist(), actor_balance_alpha=args.actor_balance_alpha,
        sampler=args.sampler, actor_weights_follow_actual_episode_probability=args.sampler=='class_anneal',
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      sb3=__import__('stable_baselines3').__version__,
                      sb3_contrib=__import__('sb3_contrib').__version__, gymnasium=gym.__version__),
    ))
    def make_env(i):
        return lambda: PrefixWaferEnv(X[train_ids], y[train_ids],
            seed=args.seed + i, wait_lambda=args.wait_lambda, wait_power=args.wait_power,
            balanced=not args.natural_sampling, random_start_fraction=args.random_start_fraction,
            correct_reward=args.correct_reward, incorrect_reward=args.incorrect_reward,
            reward_class_weights=args.reward_class_weights,sampler=args.sampler)
    vec = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    policy_kwargs = dict(features_extractor_class=PrefixCNN,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[64], vf=[64]), activation_fn=nn.Tanh)
    policy_class = ExploringPolicy if args.exploration_mix else 'MlpPolicy'
    if args.exploration_mix:
        policy_kwargs['exploration_mix'] = args.exploration_mix
    model_class = BalancedMaskablePPO if args.actor_balance_alpha else MaskablePPO
    buffer_kwargs = dict(rollout_buffer_class=BalancedRolloutBuffer,
                         rollout_buffer_kwargs=dict(class_weights=actor_weights,
                            per_transition_weights=args.sampler=='class_anneal'),
                         balance_value_loss=args.balance_value_loss) if args.actor_balance_alpha else {}
    model = model_class(policy_class, vec, learning_rate=args.lr,
        n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.epochs,
        gamma=1.0, gae_lambda=args.gae_lambda, clip_range=.2, ent_coef=args.ent_coef,
        vf_coef=.5, max_grad_norm=.5, target_kl=.03,
        policy_kwargs=policy_kwargs, seed=args.seed, device='cpu', verbose=0, **buffer_kwargs)
    if args.initial_wait_bias:
        
        
        with torch.no_grad():
            model.policy.action_net.bias[WAIT] = args.initial_wait_bias
    if args.init_checkpoint:
        checkpoint = Path(args.init_checkpoint)
        warm = BalancedMaskablePPO.load(checkpoint, device='cpu')
        model.policy.load_state_dict(warm.policy.state_dict())
        
        
        write_json(output/'warm_start.json', dict(
            source=str(checkpoint.resolve()),
            source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            optimizer='fresh Adam', classification_pretraining=False,
            description='PPO-trained weights; newly collected real on-policy data'))
        del warm
    callback = EvaluateCallback(X[monitor_ids], y[monitor_ids], output, args)
    print(f'[start] {args.output} | seed={args.seed} train={len(train_ids)} '
          f'monitor={len(monitor_ids)} params={sum(p.numel() for p in model.policy.parameters()):,} '
          f'lambda={args.wait_lambda} steps={args.steps:,}', flush=True)
    model.learn(args.steps, callback=callback)
    vec.close()
    print('[done]', args.output, flush=True)


def test(args):
    """Explicit final evaluation, once, after the recipe/checkpoint is locked."""
    torch.set_num_threads(args.threads)
    root = Path(args.output)
    result_path = root / 'official_test_once.json'
    if result_path.exists():
        raise FileExistsError(f'TEST has already been evaluated: {result_path}')
    config = json.loads((root / 'config.json').read_text())
    explicit_data = Path(args.data_dir)
    data_dir = explicit_data if (explicit_data/'wafer_TEST').is_file() else Path(config['data_dir'])
    X, y = load_split(data_dir, 'TEST')
    checkpoint_name = args.checkpoint + '_model.zip'
    model_class = BalancedMaskablePPO if config.get('actor_balance_alpha', 0) else MaskablePPO
    model = model_class.load(root / checkpoint_name, device='cpu')
    result = dict(checkpoint=checkpoint_name, test_hash=HASHES['TEST'],
                  checkpoint_sha256=hashlib.sha256((root/checkpoint_name).read_bytes()).hexdigest(),
                  seed=config['seed'], test_used_for_selection=False,
                  selected_on=('fixed_budget_before_TEST' if args.checkpoint=='last' and config['protocol']=='all_train'
                               else json.loads((root/'split_manifest.json').read_text())['monitor']))
    selection_lock = root / 'selection_lock.json'
    if selection_lock.is_file():
        result['selection_lock'] = json.loads(selection_lock.read_text())
        expected_hash = result['selection_lock'].get('checkpoint_sha256')
        if expected_hash and expected_hash != result['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after the pre-test selection lock')
    for mode in ('greedy', 'sample'):
        score, predictions = evaluate(model, X, y, config['wait_lambda'], config['wait_power'],
            stochastic=mode=='sample', seed=config['seed']+10000, return_predictions=True,
            correct_reward=config.get('correct_reward', 1.), incorrect_reward=config.get('incorrect_reward', -1.),
            reward_class_weights=config.get('reward_class_weights', (1., 1.)))
        result[mode] = score
        np.savez_compressed(root / f'test_predictions_{mode}.npz', **predictions)
    write_json(result_path, result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['train', 'test', 'check'])
    p.add_argument('--data-dir', default='data')
    p.add_argument('--output', default='runs/ppo_seed0')
    p.add_argument('--checkpoint', choices=['best', 'last'], default='best',
                   help='Explicit final-test checkpoint; choose before reading TEST results')
    p.add_argument('--protocol', choices=['holdout', 'all_train'], default='holdout')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--train-order', choices=['split', 'sorted'], default='split',
                   help='Diagnostic only: retain or sort the same training IDs; same seed is not same samples')
    p.add_argument('--steps', type=int, default=262144)
    p.add_argument('--sampler',choices=['standard','class_anneal','prefix_curriculum'],default='standard')
    p.add_argument('--max-validation-mean-t',type=float,default=0.,
                   help='Also retain a separately selected checkpoint satisfying this validation latency limit')
    p.add_argument('--save-reference-at',type=int,default=0,
                   help='Save a post-update prefix checkpoint for the training-budget control')
    p.add_argument('--n-envs', type=int, default=32)
    p.add_argument('--n-steps', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--gae-lambda', type=float, default=.95)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--ent-coef', type=float, default=.02)
    p.add_argument('--initial-wait-bias', type=float, default=0.)
    p.add_argument('--init-checkpoint', default=None,
                   help='Optional PPO-weight initialization for a recorded second stage; no supervised pretraining')
    p.add_argument('--exploration-mix', type=float, default=0.,
                   help='Optional epsilon-uniform policy family, with consistent PPO log-probabilities')
    p.add_argument('--wait-lambda', type=float, default=.001)
    p.add_argument('--wait-power', type=float, default=1/3)
    p.add_argument('--correct-reward', type=float, default=1.)
    p.add_argument('--incorrect-reward', type=float, default=-1.)
    p.add_argument('--actor-balance-alpha', type=float, default=0.,
                   help='0: original actor; 1: inverse-frequency class weights in actor/entropy loss')
    p.add_argument('--balance-value-loss', action='store_true',
                   help='Also class-weight critic squared error; requires a nonzero actor-balance-alpha')
    p.add_argument('--reward-balance-alpha', type=float, default=0.,
                   help='Class weights in ALL rewards, including WAIT; separate ablation from actor weighting')
    p.add_argument('--random-start-fraction', type=float, default=.5)
    p.add_argument('--natural-sampling', action='store_true')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--eval-every', type=int, default=16384)
    args = p.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        check_environment(*load_split(args.data_dir, 'TRAIN'))


if __name__ == '__main__':
    main()
