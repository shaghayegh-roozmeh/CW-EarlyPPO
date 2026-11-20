'Lock trained checkpoints, then evaluate validation and official TEST.'
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, matthews_corrcoef, cohen_kappa_score

from equal_budget_ppo import EqualBudgetPPO
from ppo_wafer import evaluate, load_split, write_json
from run_equal_budget_experiments import EXPERIMENT, LegacyMaskablePPO, LegacyBalancedPPO, FixedPrefixClassifier, fixed_prefix_observations

ROOT=Path(__file__).resolve().parent
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prediction_metrics(labels, predictions, times):
    cm=confusion_matrix(labels, predictions, labels=[0, 1])
    costs=np.r_[0., np.cumsum(.0001*np.arange(1, 152)**(1/3))]
    return dict(n=len(labels), evaluation='greedy', accuracy=float(accuracy_score(labels,predictions)),
        balanced_accuracy=float(balanced_accuracy_score(labels,predictions)),
        macro_f1=float(f1_score(labels,predictions,average='macro',zero_division=0)),
        weighted_f1=float(f1_score(labels,predictions,average='weighted',zero_division=0)),
        minority_f1=float(f1_score(labels,predictions,pos_label=0,zero_division=0)),
        minority_precision=float(precision_score(labels,predictions,pos_label=0,zero_division=0)),
        minority_recall=float(recall_score(labels,predictions,pos_label=0,zero_division=0)),
        majority_recall=float(recall_score(labels,predictions,pos_label=1,zero_division=0)),
        precision_by_class=precision_score(labels,predictions,labels=[0,1],average=None,zero_division=0).tolist(),
        f1_by_class=f1_score(labels,predictions,labels=[0,1],average=None,zero_division=0).tolist(),
        recall_by_class=recall_score(labels,predictions,labels=[0,1],average=None,zero_division=0).tolist(),
        mcc=float(matthews_corrcoef(labels,predictions)), cohen_kappa=float(cohen_kappa_score(labels,predictions)),
        confusion_matrix=cm.tolist(), avg_t=float(times.mean()), earliness=float(times.mean()/152),
        first_stop_fraction=float(np.mean(times==1)),
        mean_return=float(np.mean(np.where(labels==predictions,1.,-1.)-costs[times-1])),
        avg_t_by_class=[float(times[labels==c].mean()) for c in (0,1)],
        decision_time_median=float(np.median(times)), decision_time_p90=float(np.quantile(times,.9)),
        decision_time_p95=float(np.quantile(times,.95)), decision_time_max=int(times.max()),
        predicted_class_counts=np.bincount(predictions,minlength=2).tolist(),
        single_class_predictions=bool(len(np.unique(predictions))==1))


def aggregate(rows):
    keys=['accuracy','balanced_accuracy','macro_f1','weighted_f1','minority_f1',
        'minority_precision','minority_recall','majority_recall','mcc','cohen_kappa',
        'avg_t','earliness','first_stop_fraction','mean_return','decision_time_median',
        'decision_time_p90','decision_time_p95','decision_time_max']
    out={k:dict(mean=float(np.mean([r[k] for r in rows])),
                sample_sd=float(np.std([r[k] for r in rows],ddof=1))) for k in keys}
    out['avg_t_minority']=dict(mean=float(np.mean([r['avg_t_by_class'][0] for r in rows])),
        sample_sd=float(np.std([r['avg_t_by_class'][0] for r in rows],ddof=1)))
    out['avg_t_majority']=dict(mean=float(np.mean([r['avg_t_by_class'][1] for r in rows])),
        sample_sd=float(np.std([r['avg_t_by_class'][1] for r in rows],ddof=1)))
    return out


def main(models_root, data_dir, threads=2):
    torch.set_num_threads(threads)
    spec=EXPERIMENT
    static, legacy=spec['kind']=='static',spec['kind']=='legacy'
    locks=[]
    for seed in [0,1,2]:
        directory=models_root/f'seed{seed}'
        config=json.loads((directory/'config.json').read_text())
        summary=json.loads((directory/'training_summary.json').read_text())
        manifest=json.loads((directory/'split_manifest.json').read_text())
        reference=json.loads((ROOT/'PROVENANCE.json').read_text())['reference_split']
        assert manifest['train_indices']==reference['train_indices'] and manifest['monitor_indices']==reference['monitor_indices']
        assert summary['completed'] and summary['fresh_random_initialization'] if not static else summary['completed']
        assert config['seed']==seed and config['experiment']==spec['name']
        assert config['wait_lambda']==.0001 and config['reward_class_weights']==[1.,1.]
        if not static:
            assert summary['env_steps']==(1048576 if legacy else 100000)
            assert summary['actual_gradient_steps']==summary['actual_adam_state_steps'][0]
            if not legacy: assert summary['actual_gradient_steps']==100000
        checkpoints=[('selected' if static else 'final', 'selected_model.pt' if static else 'last_model.zip')]
        if legacy: checkpoints.append(('budget100k','budget100k_model.zip'))
        for stage, filename in checkpoints:
            path=directory/filename
            assert path.is_file()
            locks.append(dict(seed=seed,stage=stage,checkpoint=filename,checkpoint_sha256=sha(path),
                checkpoint_selection='minimum validation cross-entropy' if static else 'fixed predeclared interaction budget',
                TEST_used_for_selection=False))
        write_json(directory/'selection_lock.json',[r for r in locks if r['seed']==seed])
    
    
    X_train,y_train=load_split(data_dir,'TRAIN')
    X_test,y_test=load_split(data_dir,'TEST')
    runs=[]; csv_rows=[]; checks=[]
    for item in locks:
        seed,stage=item['seed'],item['stage']
        directory=models_root/f'seed{seed}'
        config=json.loads((directory/'config.json').read_text())
        summary=json.loads((directory/'training_summary.json').read_text())
        manifest=json.loads((directory/'split_manifest.json').read_text())
        path=directory/item['checkpoint']
        assert sha(path)==item['checkpoint_sha256']
        if static:
            selected=torch.load(path,map_location='cpu',weights_only=False)
            model=FixedPrefixClassifier();model.load_state_dict(selected['state_dict']);model.eval()
            optimizer_steps=selected['optimizer_steps_at_selection'];env_steps=None
        else:
            cls=(LegacyBalancedPPO if spec['alpha'] else LegacyMaskablePPO) if legacy else EqualBudgetPPO
            model=cls.load(path,device='cpu')
            env_steps=int(model.num_timesteps);optimizer_steps=int(model.actual_optimizer_steps)
            assert env_steps==(100000 if stage=='budget100k' or not legacy else 1048576)
            assert model.n_steps==64 and model.batch_size==256 and model.n_envs==32
            assert model.n_epochs==(5 if legacy else 256) and model.target_kl==(.03 if legacy else None)
            assert model.gamma==1. and model.gae_lambda==.95 and model.ent_coef==.02 and model.vf_coef==.5
            assert model.normalize_advantage and model.clip_range_vf is None
            assert model.clip_range(1)==.2 and model.max_grad_norm==.5
            assert model.policy.net_arch==dict(pi=[64],vf=[64])
            adam_steps={int(s['step']) for s in model.policy.optimizer.state.values() if 'step' in s}
            assert adam_steps=={optimizer_steps}
            if spec['kind']=='current_only':
                assert model.observation_space.shape==(1,)
                assert sum(p.numel() for p in model.policy.features_extractor.parameters())==0
            else: assert model.observation_space.shape==(152,)
        parameters=sum(p.numel() for p in (model.parameters() if static else model.policy.parameters()))
        assert parameters==(87042 if static else (516 if spec['kind']=='current_only' else 95428))
        assert all(torch.isfinite(p).all() for p in (model.parameters() if static else model.policy.parameters()))
        record=dict(**item,environment_steps=env_steps,optimizer_steps=optimizer_steps,model_parameters=parameters)
        for split,X,y in [('validation',X_train[np.asarray(manifest['monitor_indices'])],y_train[np.asarray(manifest['monitor_indices'])]),('test',X_test,y_test)]:
            if static:
                with torch.no_grad():
                    probabilities=model(torch.from_numpy(fixed_prefix_observations(X))).softmax(-1).numpy()
                predictions=dict(labels=y.copy(),predictions=probabilities.argmax(1),
                    decision_times=np.full(len(y),2,dtype=np.int64),class_probabilities=probabilities)
                original=None
            else:
                original,predictions=evaluate(model,X,y,.0001,1/3,stochastic=False,seed=seed+10000,return_predictions=True)
            labels=predictions['labels'].astype(int)
            predicted=predictions['predictions'].astype(int)
            times=predictions['decision_times'].astype(int)
            assert np.array_equal(labels,y) and np.all((times>=1)&(times<=152))
            metrics=prediction_metrics(labels,predicted,times)
            if original:
                for key in ['accuracy','balanced_accuracy','macro_f1','confusion_matrix','recall_by_class','avg_t','earliness','first_stop_fraction','mean_return','avg_t_by_class']:
                    assert np.allclose(metrics[key],original[key],rtol=1e-10,atol=1e-10),(seed,stage,split,key)
                metrics['action_counts']=original['action_counts']
                metrics['first_probs_by_class']=original['first_probs_by_class']
            prefix='' if stage in ['final','selected'] else stage+'_'
            np.savez_compressed(directory/f'{prefix}{split}_predictions_greedy.npz',**predictions)
            record[split]=metrics
            write_json(directory/f'{prefix}final_{split}_results.json',dict(seed=seed,stage=stage,
                checkpoint=item['checkpoint'],checkpoint_sha256=item['checkpoint_sha256'],
                environment_steps=env_steps,optimizer_steps=optimizer_steps,greedy=metrics))
            row=dict(experiment=spec['name'],seed=seed,stage=stage,split=split,
                checkpoint=item['checkpoint'],environment_steps=env_steps,optimizer_steps=optimizer_steps,
                model_parameters=parameters,accuracy_pct=100*metrics['accuracy'],
                balanced_accuracy_pct=100*metrics['balanced_accuracy'],macro_f1_pct=100*metrics['macro_f1'],
                weighted_f1_pct=100*metrics['weighted_f1'],minority_f1_pct=100*metrics['minority_f1'],
                minority_precision_pct=100*metrics['minority_precision'],minority_recall_pct=100*metrics['minority_recall'],
                majority_recall_pct=100*metrics['majority_recall'],mcc=metrics['mcc'],cohen_kappa=metrics['cohen_kappa'],
                mean_observations=metrics['avg_t'],earliness_pct=100*metrics['earliness'],
                first_stop_pct=100*metrics['first_stop_fraction'],minority_mean_observations=metrics['avg_t_by_class'][0],
                majority_mean_observations=metrics['avg_t_by_class'][1],mean_return=metrics['mean_return'],
                single_class_predictions=metrics['single_class_predictions'])
            csv_rows.append(row)
            print('[evaluation]',spec['name'],seed,stage,split,json.dumps(metrics),flush=True)
        if stage=='final':
            last=json.loads((directory/'monitor.jsonl').read_text().splitlines()[-1])
            assert last['final'] and last['env_steps']==env_steps
            for key in ['accuracy','balanced_accuracy','macro_f1','avg_t']:
                assert np.isclose(last[key],record['validation'][key]),(seed,key)
        checks.append(dict(seed=seed,stage=stage,checkpoint_sha256_verified=True,
            split_matches_reference=True,finite_parameters=True,parameters=parameters,
            metrics_independently_recomputed=True,actual_optimizer_counter_verified=not static))
        runs.append(record)
        del model
    main_stage='selected' if static else 'final'
    aggregates={}
    for stage in sorted({r['stage'] for r in runs}):
        aggregates[stage]={split:aggregate([r[split] for r in runs if r['stage']==stage]) for split in ['validation','test']}
    final=dict(experiment=spec,seeds=[0,1,2],train_n=800,validation_n=200,test_n=6164,
        split_seed=0,train_hash=sha(data_dir/'wafer_TRAIN'),test_hash=sha(data_dir/'wafer_TEST'),
        actual_new_training=True,main_stage=main_stage,aggregate=aggregates[main_stage]['test'],
        validation_aggregate=aggregates[main_stage]['validation'],stage_aggregates=aggregates,
        runs=runs,independent_training_runs=3,TEST_used_for_selection=False,
        official_TEST_examined_in_previous_project_rounds=True,
        single_class_runs=sum(r['test']['single_class_predictions'] for r in runs if r['stage']==main_stage))
    write_json(ROOT/'FINAL_RESULTS.json',final)
    with (ROOT/'FINAL_TEST_RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    write_json(ROOT/'VERIFICATION.json',dict(all_passed=True,checks=checks,
        checkpoint_locks_before_TEST=locks,data_hashes=dict(TRAIN=sha(data_dir/'wafer_TRAIN'),TEST=sha(data_dir/'wafer_TEST')),
        source_hashes={name:sha(ROOT/name) for name in ['ppo_wafer.py','balanced_ppo.py','equal_budget_ppo.py','run_equal_budget_experiments.py','evaluate_final.py']},
        execution='actual Python notebook cells with captured outputs and timings; no Jupyter kernel',
        every_evaluation_starts_at_t1=True,no_TEST_checkpoint_selection=True,
        generated_utc=datetime.now(timezone.utc).isoformat()))
    primary=[r for r in runs if r['stage']==main_stage]
    lines=[f"# {spec['name']}", '',spec['description'],'',
        'Three fresh independent trainings; seeds 0, 1, 2. Same original Wafer TRAIN split: 800 training / 200 validation, split_seed=0. Official TEST: 6,164 series; length 152.', '',
        '## Final results', '', '| Split | Seed | Accuracy (%) | BA (%) | Macro-F1 (%) | Mean observations |',
        '|---|---:|---:|---:|---:|---:|']
    for split in ['validation','test']:
        for r in primary:
            m=r[split]; lines.append(f"| {split} | {r['seed']} | {100*m['accuracy']:.4f} | {100*m['balanced_accuracy']:.4f} | {100*m['macro_f1']:.4f} | {m['avg_t']:.4f} |")
        a=aggregates[main_stage][split]
        vals=[f"{100*a[k]['mean']:.4f} ± {100*a[k]['sample_sd']:.4f}" for k in ['accuracy','balanced_accuracy','macro_f1']]
        lines.append(f"| {split} | Mean ± SD | {' | '.join(vals)} | {a['avg_t']['mean']:.4f} ± {a['avg_t']['sample_sd']:.4f} |")
    lines += ['', '## Controlled configuration', '', json.dumps(spec,indent=2), '',
        'PPO constants: natural class sampling, WAIT=-0.0001*t**(1/3), correct/incorrect=+1/-1, unweighted critic and environment rewards, LR=3e-4, gamma=1, GAE=.95, entropy=.02, value coefficient=.5, clip=.2, max gradient norm=.5, 32 environments x 64 steps, minibatch 256, two CPU threads per seed. Actor and critic hidden layers are 64 units each. All evaluations start at observation one and use greedy decisions.', '',
        'Legacy training: 5 PPO epochs per rollout, target_kl=.03 and 1,048,576 interactions. This is the earlier training recipe, not a change to only one isolated hyperparameter. Actual Adam updates are counted. A fixed snapshot at exactly 100,000 interactions is also evaluated; it occurs during collection, following the completed rollout at 98,304 interactions, without adding an update.', '',
        'Current training: exactly 100,000 interactions and 100,000 actual Adam steps, 256 epochs per full rollout, target_kl disabled. The final rollout is shortened as in the nine reference experiments. Old rollouts are discarded before new collection.', '',
        'Current-only ablation: actor and critic receive only scalar x_t. No historical observations, explicit clock, trainable feature encoder, substitute extractor or recurrent memory. FlattenExtractor is an identity/reshape with zero parameters; actor and critic themselves remain (516 parameters in total). Random episode initialization stays unchanged.', '',
        'Start-at-one ablation: only training episode initialization changes to t=1. The full observed-prefix CNN remains (95,428 total PPO parameters).', '',
        'Static t=2 control: a supervised network receives only [x1,x2], padded with zeros to 152 for the same CNN and 64-unit actor trunk. Two class logits replace the action head; no critic or WAIT exists (87,042 parameters). Class-weighted cross-entropy uses alpha=.5, Adam lr=3e-4/eps=1e-5, batch 256, at most 256 epochs. Minimum weighted validation cross-entropy selects the checkpoint; early stopping patience is 30 epochs. Supervised minibatch updates are reported separately and are not called environment interactions.', '',
        '## Files and reproduction', '',
        'Exactly 15 top-level items: executed notebook, five Python source files, requirements, license, FINAL_RESULTS.json, FINAL_TEST_RESULTS.csv, README.md, PROVENANCE.json, VERIFICATION.json, data/ and models/. CSV contains BOTH validation and TEST, including the legacy 100k stage. JSON contains all metrics, per-seed values and means/sample SDs. Every seed stores weights, configs, split indices, counters, validation history and predictions for both splits.', '',
        'The notebook embeds the exact source and data, and executes three fresh seed processes. Captured outputs, execution counts and timings record actual Python-cell execution; no Jupyter kernel is used. Use a new empty PROJECT for a fresh notebook rerun. Existing models are never resumed or silently overwritten.', '',
        'Single-seed terminal equivalent: python run_equal_budget_experiments.py --seed 0 --output NEW_OUTPUT/seed0 --threads 2',
        'Evaluation: python evaluate_final.py --models-root NEW_OUTPUT --data-dir data --threads 2', '',
        'All five settings were declared before these results. TEST has been examined in earlier project rounds; it is not described as untouched. No TEST result chooses checkpoints or changes settings. Archived UCR normalization is retained, matching the nine references.', '']
    (ROOT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    assert len([p for p in ROOT.iterdir() if p.name!='__pycache__'])==15, sorted(p.name for p in ROOT.iterdir())
    print('VERIFIED FINAL AGGREGATES',json.dumps(aggregates,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models-root',default='models');p.add_argument('--data-dir',default='data');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();main(Path(a.models_root),Path(a.data_dir),a.threads)
