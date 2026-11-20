"""Evaluate only last_model.zip after verifying its exact final-budget counters."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from equal_budget_ppo import EqualBudgetPPO
from ppo_wafer import evaluate, load_split, write_json


def main(args):
    torch.set_num_threads(args.threads)
    root=Path(args.models_root);out=Path(args.output)
    if out.exists():raise FileExistsError('Choose a fresh evaluation output directory')
    items=[]
    for seed in [0,1,2]:
        directory=root/f'seed{seed}'
        config=json.loads((directory/'config.json').read_text())
        summary=json.loads((directory/'training_summary.json').read_text())
        assert config['seed']==seed and config['sampler']=='standard' and config['actor_balance_alpha']==0.
        assert config['wait_lambda']==.00001 and config['wait_power']==1/3
        assert summary['env_steps']==summary['actual_gradient_steps']==100000
        checkpoint=directory/'last_model.zip'
        items.append(dict(seed=seed,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            checkpoint_selection='fixed_final_budget',test_used_for_selection=False))
    out.mkdir(parents=True)
    write_json(out/'final_checkpoint_lock.json',items)
    X,y=load_split(args.data_dir,'TEST')
    results=[]
    for item in items:
        model=EqualBudgetPPO.load(item['checkpoint'],device='cpu')
        assert model.num_timesteps==model.actual_optimizer_steps==100000
        assert hashlib.sha256(Path(item['checkpoint']).read_bytes()).hexdigest()==item['checkpoint_sha256']
        record=dict(**item,environment_steps=100000,optimizer_steps=100000)
        metrics,predictions=evaluate(model,X,y,.00001,1/3,stochastic=False,
            seed=item['seed']+10000,return_predictions=True)
        record['greedy']=metrics
        np.savez_compressed(out/f"seed{item['seed']}_greedy_predictions.npz",**predictions)
        results.append(record)
        print('seed',item['seed'],'final TEST',record['greedy'],flush=True)
    write_json(out/'final_test_results.json',results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models-root',default='models');p.add_argument('--data-dir',default='data')
    p.add_argument('--output',required=True);p.add_argument('--threads',type=int,default=2)
    main(p.parse_args())
