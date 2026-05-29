# LagFormer

A Transformer-based Neural ODE model for inferring directed effective connectivity from fMRI time series. LagFormer uses lag-aware multi-head attention to identify causal relationships between brain regions across multiple temporal offsets, then integrates those relationships through a Neural ODE to produce predictions.

## Architecture

![Lag Attention Architecture](Lag%20Attention.svg)

Inputs feed into two parallel branches. The upper branch processes lagged copies of the signal (lag = 0…L), projecting each through a Linear layer to form query/key pairs. A Dot Product operation computes attention matrices at each lag, stacked into a lag-indexed tensor. A Weighted Sum collapses the lag dimension, yielding a single Causal Graph Adjacency Matrix. That matrix drives a Neural ODE (lower branch) which integrates the dynamics forward (and optionally backward) to produce Predicted Outputs.

**Key modules:**

| File | Role |
|------|------|
| `model.py` | `Model`: conv encoder + positional encoding + attention + Neural ODE |
| `layers.py` | `LaggingMultiHeadAttention`, `PositionwiseFeedForward`, `PositionalEncoding` |
| `train_sanch.py` | Training loop, experiment runner, CLI entry point |
| `data_sanch.py` | Dataset loading, sliding-window preparation |
| `utils/losses.py` | MSE + sparsity + structure + transitivity + degree regularizers |
| `utils/plotter.py` | Lag-attention heatmaps, prediction plots, metric curves |
| `optim.py` | Transformer-style warmup LR scheduler |

## Dataset

Simulated fMRI BOLD time series from:

> Sanchez-Romero, R., & Cole, M. W. (2021). *Estimating feedforward and feedback effective connections from fMRI time series: Assessments of statistical methods.* Network Neuroscience, 5(2), 549–564.

Raw data on OpenNeuro:
**[ds000003 v00001](https://openneuro.org/datasets/ds000003/versions/00001)**

The `dataset/sanch/` directory contains pre-processed, FSL-filtered BOLD signals organised by network topology:

```
dataset/sanch/
├── Network1_amp/          # feedforward network, amplitude modulation
│   ├── SessionsConcatenated.txt
│   ├── ground_truth.txt
│   └── data_fslfilter/    # per-subject BOLDfslfilter_XX.txt
├── Network2_amp/
├── Network3_amp/
├── Network4_amp/
├── Network5_amp/
├── Network5_cont/         # contrast variants
├── Network5_cont_p3n7/
├── Network5_cont_p7n3/
├── Network6_amp/
├── Network6_cont/
├── Network7_amp/
├── Network7_cont/
├── Network8_amp_cont/
├── Network8_cont_amp/
├── Network8_amp_amp/
├── Network9_amp_amp/
├── Network9_cont_amp/
└── Network9_amp_cont/
```

`ground_truth.txt` provides the ground-truth adjacency matrix for Networks 1–4, used to compute precision, recall, F1, accuracy, and SHD metrics.

## Requirements

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate lagformer
```

Tested with Python 3.13, PyTorch 2.11.0, NumPy 2.3, SciPy 1.16. For GPU support, replace the `torch` pip entry with the appropriate CUDA wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

## Usage

### Train on a single network index

```bash
python train_sanch.py --index 1 --lag 10 --epoch 301 --d_model 16 --n_head 2 \
    --d_inner_hid 64 --pred_len 5 --batch_size 32 --dropout 0.2 --alpha 0.8
```

### Train across all network indices

```bash
python train_sanch.py --run_all --lag 10 --epoch 301 --alpha 0.8
```

### Key hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--lag` | `10` | Lookback lag length (or comma-separated list) |
| `--d_model` | `16` | Embedding dimension |
| `--n_head` | `2` | Attention heads |
| `--d_inner_hid` | `64` | FFN hidden dimension |
| `--pred_len` | `5` | Forward prediction steps |
| `--alpha` | `0.8` | L1 sparsity penalty weight |
| `--beta` | `0.0` | Structure penalty weight |
| `--gamma` | `0.0` | Transitivity penalty weight |
| `--delta` | `0.0` | Degree balance penalty weight |
| `--epoch` | `301` | Training epochs |
| `--batch_size` | `32` | Batch size |
| `--dropout` | `0.2` | Dropout rate |

Checkpoints are saved to `checkpoints/` and logs to `logs/`.

## Evaluation Metrics

`utils/utils.py:cal_metrics` computes against ground-truth adjacency:

- **Precision** — fraction of predicted edges that are true
- **Recall** — fraction of true edges recovered
- **F1** — harmonic mean of precision and recall
- **Accuracy** — overall edge classification accuracy
- **SHD** (Structural Hamming Distance) — edit distance between predicted and true graphs

## References

Sanchez-Romero, R., & Cole, M. W. (2021). Estimating feedforward and feedback effective connections from fMRI time series: Assessments of statistical methods. *Network Neuroscience*, 5(2), 549–564.
