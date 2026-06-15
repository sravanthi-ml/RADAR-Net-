# RADAR-Net: Reinforcement-Guided Adversarially Robust Detection and Resilience Network for Rumor Verification on Social Media

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-blue)](.)
[![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Datasets](https://img.shields.io/badge/Datasets-PHEME%20%7C%20Twitter15%20%7C%20Twitter16%20%7C%20Weibo-purple)]()

</div>

---

## Overview

**RADAR-Net** is the first unified framework for adversarially robust rumor detection that simultaneously addresses:

- ✅ **Multi-level adversarial attacks** — token, sentence, and propagation-graph levels
- ✅ **Adaptive defense** — reinforcement learning selects the optimal defense strategy per input
- ✅ **Formal robustness certification** — per-sample Lipschitz-bounded guarantees (first in the field)

> **Key result:** 90.3% clean accuracy · 82.5% adversarial accuracy · 67.4% CRR at ε=0.05 · 15.8 ms inference — a **54.9% reduction** in adversarial accuracy drop over the strongest baseline (HAT4RD), with **zero efficiency trade-off**.

---

## Architecture

```
Social Media Post (text thread + propagation graph)
        │
   ┌────┴─────┐
   ▼          ▼
  CTE        LPE
(DistilBERT) (2-layer GCN)
H_text∈ℝ⁷⁶⁸  H_graph∈ℝ¹²⁸
   │          │
   └────┬─────┘
        ▼
       FFM  ← bilinear gating
   z ∈ ℝ²⁵⁶
   ┌────┼────┐
   ▼    ▼    ▼
 HATM  RLPN  CRL
  │     │     │
  └─────┴─────┘
        ▼
  ŷ + ε_cert certificate
```

| Module | Function | Key Mechanism |
|--------|----------|---------------|
| **CTE** | Text encoding | 6-layer DistilBERT, mean pooling, spectral norm |
| **LPE** | Graph encoding | 2-layer GCN on propagation tree |
| **FFM** | Fusion | Bilinear gating — adapts to graph density |
| **HATM** | Adversarial training | PGD (token) + gradient ascent (sentence) + edge toggle (graph) |
| **RLPN** | Adaptive defense | MDP, REINFORCE, 5 defense actions, ε-greedy |
| **CRL** | Certification | Spectral normalisation, Lipschitz bound, ε_cert = margin/2 |

---

## Results

### Clean Accuracy (Table 4)

| Model | PHEME | Twitter15 | Twitter16 | Weibo | Avg | ms |
|-------|-------|-----------|-----------|-------|-----|----|
| BiGCN | 82.1 | 86.2 | 87.6 | 81.4 | 84.3 | 9.2 |
| DGTR | 86.5 | 89.7 | 91.2 | 85.7 | 88.3 | 41.1 |
| HAT4RD | 85.2 | 87.9 | 89.8 | 84.1 | 86.8 | 22.6 |
| **RADAR-Net** | **88.9** | **91.4** | **93.3** | **87.5** | **90.3** | **15.8** |

### Adversarial Accuracy (Table 5, avg over 4 datasets)

| Model | FGSM | PGD | TextFooler | BERT-Atk | Graph-Atk | Avg | ACC-Drop |
|-------|------|-----|------------|----------|-----------|-----|----------|
| HAT4RD | 72.6 | 67.3 | 69.4 | 66.2 | 71.8 | 69.5 | 17.3 |
| **RADAR-Net** | **84.1** | **79.6** | **82.3** | **80.8** | **85.7** | **82.5** | **7.8** |
| Δ vs HAT4RD | ↑11.5 | ↑12.3 | ↑12.9 | ↑14.6 | ↑13.9 | ↑13.0 | ↓9.5 (55%↓) |

### Certified Robustness (Table 6)

| Model | CRR @ ε=0.05 |
|-------|--------------|
| BERT-base + Smoothing | 41.8% |
| HAT4RD | 34.2% |
| **RADAR-Net** | **67.4%** |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/RADAR-Net.git
cd RADAR-Net

# Create virtual environment
python -m venv radarnet_env
source radarnet_env/bin/activate   # Windows: radarnet_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+, CUDA 11.8+ (recommended), 16GB+ RAM, NVIDIA GPU with 24GB+ VRAM (A100 80GB used in paper).

---

## Datasets

RADAR-Net is evaluated on four benchmark datasets. Download instructions are in [`data/README.md`](data/README.md).

| Dataset | Language | Posts | Platform | Avg Replies |
|---------|----------|-------|----------|-------------|
| PHEME | English | 6,425 | Twitter | 14.2 |
| Twitter15 | English | 1,490 | Twitter | 223.4 |
| Twitter16 | English | 818 | Twitter | 251.7 |
| Weibo | Chinese | 4,664 | Weibo | 35.6 |

All datasets are split 70/10/20 (train/val/test) with stratified sampling.

```bash
# Download and preprocess all datasets
bash scripts/download_datasets.sh
python scripts/preprocess.py --dataset all
```

---

## Quick Start

### Training

```bash
# Train on PHEME (full 3-phase protocol)
python scripts/train.py \
    --dataset pheme \
    --phase all \
    --device cuda:0

# Train on Twitter15
python scripts/train.py \
    --dataset twitter15 \
    --phase all \
    --device cuda:0

# Train phase by phase
python scripts/train.py --dataset pheme --phase 1   # backbone + HATM
python scripts/train.py --dataset pheme --phase 2   # RLPN (frozen backbone)
python scripts/train.py --dataset pheme --phase 3   # joint fine-tuning
```

### Evaluation

```bash
# Evaluate on all attack scenarios
python scripts/evaluate.py \
    --dataset pheme \
    --checkpoint checkpoints/pheme_best.pt \
    --attacks all

# Evaluate single attack
python scripts/evaluate.py \
    --dataset twitter16 \
    --checkpoint checkpoints/twitter16_best.pt \
    --attacks pgd

# Compute certified robustness rate
python scripts/evaluate.py \
    --dataset weibo \
    --checkpoint checkpoints/weibo_best.pt \
    --crr \
    --eps_threshold 0.05
```

### Reproduce Paper Results

```bash
# Reproduce all tables in one command (requires all 4 datasets downloaded)
bash scripts/reproduce_all.sh

# Reproduce specific table
python scripts/reproduce.py --table 4    # clean accuracy
python scripts/reproduce.py --table 5    # adversarial accuracy
python scripts/reproduce.py --table 6    # certified robustness
python scripts/reproduce.py --table 7    # ablation
python scripts/reproduce.py --table 8    # sensitivity
python scripts/reproduce.py --table 9    # cross-dataset transfer
```

---

## Configuration

All hyperparameters are in [`configs/radar_net.yaml`](configs/radar_net.yaml):

```yaml
model:
  d_model: 768          # CTE hidden dimension
  d_graph: 128          # LPE output dimension
  d_fuse: 256           # FFM fusion dimension
  sigma_target: 1.0     # CRL spectral norm target

hatm:
  epsilon_token: 0.1    # L2 perturbation radius
  pgd_steps: 3          # PGD steps (training)
  lambda_1: 0.4         # token-level weight
  lambda_2: 0.3         # sentence-level weight
  lambda_3: 0.3         # graph-level weight
  lambda_H: 0.5         # total HATM weight
  top_k_edges: 5        # graph edge perturbations

rlpn:
  lr: 5.0e-4
  gamma: 0.95
  eps_start: 1.0
  eps_end: 0.05
  buffer_capacity: 10000

training:
  phase1_epochs: 50
  phase2_epochs: 20
  phase3_epochs: 10
  batch_size: 32
  lr_transformer: 2.0e-5
  lr_gcn: 1.0e-3
  grad_clip: 1.0
  early_stop_patience: 5
```

---

## Project Structure

```
RADAR-Net/
├── README.md
├── requirements.txt
├── LICENSE
├── configs/
│   └── radar_net.yaml          # All hyperparameters
├── data/
│   ├── README.md               # Dataset download instructions
│   └── scripts/
│       ├── download.sh
│       └── preprocess.py
├── src/
│   ├── models/
│   │   ├── radar_net.py        # Main model class
│   │   ├── cte.py              # Compact Transformer Encoder
│   │   ├── lpe.py              # Lightweight Propagation Encoder
│   │   ├── ffm.py              # Feature Fusion Module
│   │   ├── rlpn.py             # RL Policy Network
│   │   └── crl.py              # Certified Robustness Layer
│   ├── training/
│   │   ├── trainer.py          # 3-phase training loop
│   │   └── hatm.py             # HATM adversarial augmentation
│   ├── attacks/
│   │   ├── fgsm.py
│   │   ├── pgd.py
│   │   ├── textfooler.py
│   │   ├── bert_attack.py
│   │   └── graph_attack.py     # GAFSI-based
│   ├── defense/
│   │   ├── actions.py          # 5 RLPN defense actions
│   │   └── certify.py          # CRR computation
│   └── utils/
│       ├── data_loader.py
│       ├── graph_utils.py
│       └── metrics.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── reproduce.py
│   ├── reproduce_all.sh
│   └── download_datasets.sh
├── results/
│   ├── tables/                 # CSV files of all paper tables
│   └── figures/                # High-resolution figures (600 DPI)
├── tests/
│   └── test_components.py
└── docs/
    └── architecture.md
```

---

## Formal Guarantees

RADAR-Net provides **per-sample certified robustness**:

```
ε_cert(X, G) = (F_θ[c*] − max_{c≠c*} F_θ[c]) / 2
```

For any perturbation `‖(δ, ΔA)‖₂ ≤ ε_cert`, the classification is **provably unchanged** — regardless of the attack algorithm used (Theorem 2).

This guarantee holds because every weight matrix satisfies `σ(W_l) ≤ 1` (spectral normalisation), making the entire network 1-Lipschitz (Lemma 1).

---

## Citation

If you use RADAR-Net in your research, please cite:

```bibtex
@article{radarnet2025,
  title     = {{RADAR-Net}: Reinforcement-Guided Adversarially Robust Detection
               and Resilience Network for Rumor Verification on Social Media},
  author    = {[Authors]},
  journal   = {[Journal]},
  year      = {2025},
  note      = {Under Review}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

We thank the authors of [DistilBERT](https://arxiv.org/abs/1910.01108), [BiGCN](https://arxiv.org/abs/2001.06362), [HAT4RD](https://www.mdpi.com/1424-8220/22/17/6652), [GAFSI](https://dl.acm.org/doi/10.1145/3637528.3671902), and [FreeLB](https://arxiv.org/abs/1902.03932) for their publicly available implementations.

---

<div align="center">
  <sub>Questions? Open an issue or contact [author email].</sub>
</div>
