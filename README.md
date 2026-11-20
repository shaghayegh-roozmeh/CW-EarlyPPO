# Early Classification of Industrial Sensor Signals with Class-Weighted Proximal Policy Optimization

This repository contains the code and experimental results for our study of **class-weighted Proximal Policy Optimization (PPO)** for early classification of industrial sensor signals on the UCR Wafer dataset.

Early classification requires a model to decide both *what* a signal represents and *when* it has seen enough to make that decision. This is especially challenging when one class is much less common than the other: a policy can achieve high overall accuracy while missing many minority-class examples. We study whether weighting the PPO policy and entropy losses can improve minority-class recognition without changing the original class distribution or the reward function.

The agent observes a growing prefix of a sensor signal and can either wait for another observation or predict one of two classes. Our experiments compare class-weighting strengths, waiting penalties, observation settings, and PPO training configurations.

## Repository structure

All paths below are relative to the repository root. The examples assume that the shared dataset folder is named `data/`; adjust the paths if your copy uses a different name or location.

| Path | Contents |
|---|---|
| `data/` | Shared Wafer dataset: `wafer_TRAIN` and `wafer_TEST` |
| `experiments/PPO_Standard_Mild_100k_Final_3Seeds/` | Main experiment with moderate class weighting (`alpha = 0.5`) |
| `experiments/ppo_alpha0/` | No class weighting |
| `experiments/ppo_alpha1/` | Stronger class weighting |
| `experiments/ppo_alpha*_wait*/` | Waiting-penalty experiments |
| `experiments/ppo_alpha05_start1/` | Episodes start at the first observation |
| `experiments/ppo_alpha05_current_only/` | Current observation only |
| `experiments/ppo_legacy_updates_alpha*/` | Alternative PPO update settings |
| `results/` | Per-seed and summary CSV tables |

Each experiment has its own Python files, requirements, executed notebook, and reported results. Source files are kept with their experiments to preserve the original imports. **The dataset is stored once in the shared folder and is no longer duplicated inside each experiment directory.**

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

The experiments were run with **Python 3.12**. From the repository root, enter the main experiment directory and install its dependencies:

```bash
cd experiments/PPO_Standard_Mild_100k_Final_3Seeds
python -m pip install -r requirements_equal_budget.txt
```

### Configure the dataset path

The shared dataset folder must contain the original files named **`wafer_TRAIN`** and **`wafer_TEST`**. Keep their names and contents unchanged; the data loader checks the original file hashes.

Some scripts and notebook cells still use the original experiment-local `data/` path. Before running an experiment, choose one of the following approaches:

1. **Use the shared dataset folder.** For scripts that accept `--data-dir`, pass the path explicitly. From an experiment directory directly inside `experiments/`, the shared folder at the repository root is `../../data`. For scripts with a fixed data path, update that path in the code as described below.
2. **Use an experiment-local copy.** Create a `data/` directory inside the experiment you want to run and copy `wafer_TRAIN` and `wafer_TEST` into it. The scripts can then use their original local paths. For evaluation with this arrangement, use `--data-dir data`.

An absolute path to the shared dataset folder can also be used. Relative paths passed on the command line are resolved from the directory where the command is run.

### Evaluate the saved models

The main experiment includes the final checkpoints for all three seeds. The evaluation script accepts `--data-dir`, so the shared dataset can be used directly without retraining:

```bash
python evaluate_final.py --models-root models --data-dir ../../data --output reevaluation_01
```

This command assumes that you are still in `experiments/PPO_Standard_Mild_100k_Final_3Seeds/` and that the shared dataset is in the repository-root `data/` folder. Use a new output directory for each evaluation; the script will not overwrite an existing one.

### Train a new model

The training launcher `run_equal_budget_experiments.py` sets the dataset path inside the code and **does not accept a `--data-dir` command-line argument**.

To train using the shared repository-root dataset, find the following setting in the experiment's `run_equal_budget_experiments.py`:

```python
data_dir=str(ROOT / 'data')
```

Replace it with:

```python
data_dir=str(ROOT.parent.parent / 'data')
```

Here, `ROOT` is the experiment directory. If you chose the experiment-local copy described above, keep the original setting.

After configuring the dataset path, train the main configuration from scratch with seed 0:

```bash
python run_equal_budget_experiments.py \
  --single G_stable_weighted_equal_budget \
  --seed 0 \
  --output fresh_run_seed0
```

Change the seed and output directory to run the other seeds. Apply the same dataset-path setup separately to any other experiment you run.

### Run the notebooks

The experiment folders include executed notebooks with recorded outputs. Before enabling training or evaluation cells, check the notebook's dataset path and point it to the shared folder, or provide an experiment-local copy of the two dataset files.

Some notebook cells write Python scripts or recreate local data files. Check the paths in those cells as well: rerunning a script-writing cell can restore the original experiment-local dataset path and overwrite a manual edit to the corresponding Python file. Run notebooks in a separate working copy if you want to preserve the recorded experiment files.

The other ablation folders include their code, notebooks, and reported results, but **not** their trained model checkpoints. Those models must be trained again to reevaluate them directly, using the shared dataset or an experiment-local copy as described above.

## Dataset and evaluation

We use the **Wafer** dataset from the UCR 2015 Time Series Classification Archive. The original training split has 1,000 series, of which 800 are used for training and 200 for validation; the official test split contains 6,164 series. Each series has 152 observations. The experiments use the supplied archive representation and report metrics separately for the minority and majority classes.

The main comparison keeps the sampling procedure and reward function fixed while changing the class weights applied to the policy and entropy terms. The value loss remains unweighted. Comparisons with published methods in `results/` are provided for context; they may use different training protocols or evaluation settings.

## Notes

The implementation builds on Stable-Baselines3 and `sb3-contrib`; the relevant upstream license notice is included in the experiment directories. Please also check the UCR archive's terms when redistributing the dataset. The repository is intended as a companion to the paper; it does not imply that all approaches were trained under identical computational budgets.
