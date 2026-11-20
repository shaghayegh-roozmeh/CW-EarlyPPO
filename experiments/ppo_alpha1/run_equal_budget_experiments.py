'Two frozen recipes, three seeds; exactly 100k environment and optimizer steps.'
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from stable_baselines3.common.vec_env import DummyVecEnv

from balanced_ppo import BalancedRolloutBuffer, inverse_frequency_weights
from equal_budget_ppo import EqualBudgetPPO, UpdateEvaluateCallback
from ppo_wafer import PrefixCNN, PrefixWaferEnv, HASHES, check_environment, load_split, write_json

ROOT = Path(__file__).resolve().parent
ARMS = {
    'F_fast_prefix_equal_budget': dict(sampler='prefix_curriculum', alpha=0.),
    'G_stable_weighted_equal_budget': dict(sampler='standard', alpha=1.),
}
SOURCES = ['ppo_wafer.py','balanced_ppo.py','equal_budget_ppo.py','run_equal_budget_experiments.py']


def train_one(arm, seed, steps=100000, output=None, target_kl=None, threads=2, eval_every=5000):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    X,y = load_split(ROOT/'data','TRAIN')
    check_environment(X,y)
    train_ids,val_ids = train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=0)
    settings = ARMS[arm]
    directory = Path(output) if output else ROOT/'equal_budget_runs'/f'{arm}_seed{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'config.json').exists():raise FileExistsError(directory)
    alpha = settings['alpha']
    weights = inverse_frequency_weights(y[train_ids],alpha)
    args = argparse.Namespace(arm=arm, mode='train',protocol='holdout',seed=seed,split_seed=0,
        train_order='split',steps=steps,optimizer_steps=steps,natural_sampling=True,
        n_envs=32,n_steps=64,batch_size=256,epochs=256,gae_lambda=.95,lr=3e-4,
        ent_coef=.02,initial_wait_bias=0.,init_checkpoint=None,exploration_mix=0.,
        wait_lambda=.0001,wait_power=1/3,correct_reward=1.,incorrect_reward=-1.,
        actor_balance_alpha=alpha,balance_value_loss=False,reward_balance_alpha=0.,
        reward_class_weights=[1.,1.],random_start_fraction=.5,threads=threads,
        sampler=settings['sampler'],max_validation_mean_t=5.73,save_reference_at=0,
        eval_every=eval_every,eval_every_unit='actual_optimizer_steps',target_kl=target_kl,
        data_dir=str(ROOT/'data'),output=str(directory),
        exact_budget=True,algorithm='clipped masked PPO; high reuse of each fresh rollout')
    write_json(directory/'config.json',vars(args))
    write_json(directory/'split_manifest.json',dict(dataset='original UCR2015 Wafer',
        train_hash=HASHES['TRAIN'],train_indices=train_ids.tolist(),monitor_indices=val_ids.tolist(),
        monitor='independent_validation',test_used_for_selection=False,
        observation='right-zero-padded actual prefix; no explicit clock, label or future',
        actor_class_weights=weights.tolist(),actor_balance_alpha=alpha,
        sampler=settings['sampler'],reward=dict(correct=1.,incorrect=-1.,wait_lambda=.0001,
            wait_power=1/3,class_weights=[1.,1.]),
        classification_pretraining=False,auxiliary_classification_loss=False,
        initialization='from scratch',
        versions=dict(python=platform.python_version(),torch=torch.__version__,
            sb3=__import__('stable_baselines3').__version__,sb3_contrib=__import__('sb3_contrib').__version__,
            gymnasium=__import__('gymnasium').__version__)))
    def make_env(i):
        return lambda:PrefixWaferEnv(X[train_ids],y[train_ids],seed=seed+i,
            wait_lambda=args.wait_lambda,wait_power=args.wait_power,balanced=False,
            random_start_fraction=.5,sampler=args.sampler)
    vec = DummyVecEnv([make_env(i) for i in range(32)])
    model = EqualBudgetPPO('MlpPolicy',vec,learning_rate=args.lr,n_steps=64,batch_size=256,
        n_epochs=256,gamma=1.,gae_lambda=.95,clip_range=.2,ent_coef=.02,vf_coef=.5,
        max_grad_norm=.5,target_kl=target_kl,seed=seed,device='cpu',verbose=0,
        rollout_buffer_class=BalancedRolloutBuffer,rollout_buffer_kwargs=dict(class_weights=weights),
        policy_kwargs=dict(features_extractor_class=PrefixCNN,
            features_extractor_kwargs=dict(features_dim=128),net_arch=dict(pi=[64],vf=[64]),
            activation_fn=nn.Tanh))
    assert sum(p.numel() for p in model.policy.parameters())==95428
    callback = UpdateEvaluateCallback(X[val_ids],y[val_ids],directory,args)
    print(f'[start] {arm} seed={seed} interactions={steps} optimizer_steps={steps} target_kl={target_kl}',flush=True)
    try:model.learn_exact(steps,steps,callback)
    finally:vec.close()
    print('[completed]',directory,flush=True)


def main(workers=4):
    output=ROOT/'equal_budget_runs';output.mkdir(exist_ok=True)
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    plan=dict(arms=ARMS,seeds=[0,1,2],train_n=800,validation_n=200,split_seed=0,
        interactions=100000,actual_optimizer_steps=100000,
        primary_checkpoint='highest validation accuracy, ties highest return',
        secondary_checkpoint='same subject to mean validation t <= 5.73',
        evaluation_every=5000,evaluation_unit='actual optimizer steps',
        constants=dict(n_envs=32,n_steps=64,batch_size=256,lr=.0003,gamma=1.,gae_lambda=.95,
            ent_coef=.02,vf_coef=.5,clip_range=.2,max_grad_norm=.5,parameters=95428,threads=2),
        target_kl=None,previous_target_kl=.03,
        reason_target_kl_changed='Exact 100k genuine updates require allowing high rollout reuse; original KL early stop would cap updates',
        epoch_rule='256 full epochs per 2048-transition rollout; last 1696-transition rollout gets 1696 real steps with a partial final epoch',
        interaction_rule='48 rollouts of 2048 transitions and one of 1696; no SB3 overshoot',
        actor_alpha_schedule='constant for entire training',
        initialization='all from scratch; incumbent 98.70% weights untouched',
        old_rollouts_replayed=False,old_log_prob_relabelled=False,
        TEST_used_for_selection=False,official_TEST_seen_in_previous_rounds=True,
        not_identical_to_Martinez='same headline budgets, different algorithm, minibatch sizes and validation protocol',
        source_hashes=hashes)
    path=ROOT/'equal_budget_experiment_plan.json'
    if path.exists():assert json.loads(path.read_text())==plan,'Frozen plan changed'
    else:write_json(path,plan)
    pending=[]
    for seed in [0,1,2]:
        for arm in ARMS:
            name=f'{arm}_seed{seed}';directory=output/name
            if (directory/'training_summary.json').exists():
                summary=json.loads((directory/'training_summary.json').read_text())
                assert summary['completed'] and summary['env_steps']==100000 and summary['actual_gradient_steps']==100000
                continue
            pending.append((name,[sys.executable,'-u',str(Path(__file__).resolve()),'--single',arm,'--seed',str(seed)]))
    active=[]
    try:
        while pending or active:
            while pending and len(active)<workers:
                name,command=pending.pop(0)
                log=(output/f'{name}.log').open('w')
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                active.append((name,process,log));print('Started',name,flush=True)
            for item in active[:]:
                name,process,log=item;status=process.poll()
                if status is None:continue
                log.close();active.remove(item);print('Finished',name,'exit',status,flush=True)
                if status:raise RuntimeError(f'{name} failed; inspect its log')
            if active:time.sleep(1)
    finally:
        for _,process,log in active:
            if process.poll() is None:process.terminate()
            log.close()
    print('All six trainings completed. This runner never loads TEST.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4)
    p.add_argument('--single',choices=list(ARMS));p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=100000);p.add_argument('--output')
    p.add_argument('--target-kl',type=float);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--eval-every',type=int,default=5000)
    a=p.parse_args()
    if a.single:train_one(a.single,a.seed,a.steps,a.output,a.target_kl,a.threads,a.eval_every)
    else:main(a.workers)
