# Class-Weighted PPO for Early Time-Series Classification

This repository contains the code and experimental results for our study of **class-weighted Proximal Policy Optimization (PPO)** for early time-series classification on the UCR Wafer dataset.

Early classification requires a model to decide both *what* a signal represents and *when* it has seen enough to make that decision. This is especially challenging when one class is much less common than the other: a policy can achieve high overall accuracy while missing many minority-class examples. We study whether weighting the PPO policy and entropy losses can improve minority-class recognition without changing the original class distribution or the reward function.

The agent observes a growing prefix of a sensor signal and can either wait for another observation or predict one of two classes. Our experiments compare class-weighting strengths, waiting penalties, observation settings, and PPO training configurations.

## Repository structure

```text
CW-EarlyPPO/
├── experiments/
│   ├── PPO_Standard_Mild_100k_Final_3Seeds/   # Main experiment (alpha = 0.5)
│   ├── ppo_alpha0/                            # No class weighting
│   ├── ppo_alpha1/                            # Stronger class weighting
│   ├── ppo_alpha*_wait*/                      # Waiting-penalty experiments
│   ├── ppo_alpha05_start1/                    # Episodes start at the first observation
│   ├── ppo_alpha05_current_only/              # Current observation only
│   └── ppo_legacy_updates_alpha*/            # Alternative PPO update settings
└── results/                                   # Per-seed and summary CSV tables
```

Each experiment has its own Python files, requirements, data, executed notebook, and final results. We keep the source files together rather than moving them into a shared package, so the original imports and relative data paths are preserved.

## Main results

The table reports the **official Wafer test-set** results averaged over three independent runs (seeds 0, 1, and 2). Each run is evaluated at its **final** checkpoint, after 100,000 environment interactions and 100,000 actual optimizer steps.

| Metric | PPO, alpha = 0 | Class-weighted PPO, alpha = 0.5 |
|---|---:|---:|
| Accuracy (%) | 97.18 | **97.73** |
| Balanced accuracy (%) | 87.96 | **90.93** |
| Minority-class recall (%) | 76.19 | **82.26** |
| Mean observations before prediction | **1.53** | 1.83 |

Mild class weighting improves minority-class recognition in these runs, with a modest increase in the average number of observations used. Detailed results for individual seeds and other configurations are in `results/` and the corresponding experiment folders.

## Getting started

The experiments were run with **Python 3.12**. Start from the main experiment directory:

```bash
cd experiments/PPO_Standard_Mild_100k_Final_3Seeds
python -m pip install -r requirements_equal_budget.txt
```

### Evaluate the saved models

The main experiment includes the final checkpoints for all three seeds. You can evaluate them without retraining:

```bash
python evaluate_final.py --models-root models --data-dir data --output reevaluation_01
```

Use a new output directory for each evaluation; the script will not overwrite an existing one.

### Train a new model

To train the main configuration from scratch with seed 0:

```bash
python run_equal_budget_experiments.py \
  --single G_stable_weighted_equal_budget \
  --seed 0 \
  --output fresh_run_seed0
```

Change the seed and output directory to run the other seeds. The experiment folders also include executed notebooks with recorded outputs. Some notebooks can write local source or data files when their training cells are enabled, so run them in a separate working copy if you want to preserve this archive unchanged.

The other ablation folders include their code, data, notebooks, and reported results, but **not** their trained model checkpoints. Those models must be trained again to reevaluate them directly.

## Dataset and evaluation

We use the **Wafer** dataset from the UCR 2015 Time Series Classification Archive. The original training split has 1,000 series, of which 800 are used for training and 200 for validation; the official test split contains 6,164 series. Each series has 152 observations. The experiments use the supplied archive representation and report metrics separately for the minority and majority classes.

The main comparison keeps the sampling procedure and reward function fixed while changing the class weights applied to the policy and entropy terms. The value loss remains unweighted. Comparisons with published methods in `results/` are provided for context; they may use different training protocols or evaluation settings.

## Notes

The implementation builds on Stable-Baselines3 and `sb3-contrib`; the relevant upstream license notice is included in the experiment directories. Please also check the UCR archive's terms when redistributing the dataset. The repository is intended as a companion to the paper; it does not imply that all approaches were trained under identical computational budgets.
