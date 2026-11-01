# VGG6 on CIFAR-10 — CS6886W Assignment 1

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-FFBE00?logo=weightsandbiases&logoColor=black)](https://wandb.ai/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cs25m506/vgg6-cifar10-assignment/blob/main/notebooks/CS6886W_A1.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Course:** CS6886W · Deep Learning Systems  
> **Assignment:** A1 — Training VGG6 on CIFAR-10 with activation & optimiser experiments  
> **Tracking:** All runs logged to [Weights & Biases](https://wandb.ai/) — project `CS6886W-A1-VGG6-CIFAR10`

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Repository Structure](#-repository-structure)
3. [VGG6 Architecture](#-vgg6-architecture)
4. [Installation — Local](#-installation--local)
5. [Running on Google Colab](#-running-on-google-colab)
6. [Experiments](#-experiments)
   - [Q1 — Baseline](#q1--baseline)
   - [Q2(a) — Activation Functions](#q2a--activation-functions)
   - [Q2(b) — Optimisers](#q2b--optimisers)
   - [Q2(c) — Hyperparameters](#q2c--hyperparameters)
   - [Q3 — Analysis & Plots](#q3--analysis--plots)
7. [W&B Dashboard & Plots](#-wb-dashboard--plots)
8. [Results](#-results)
9. [Project Configuration](#-project-configuration)
10. [Contributing](#-contributing)

---

## 🔍 Overview

This repository implements a **VGG6** convolutional neural network trained on **CIFAR-10** to systematically compare:

| Question | What is varied | # Runs |
|----------|---------------|--------|
| Q1       | Best single configuration (ReLU + Adam) | 1 |
| Q2(a)    | 5 activation functions | 5 |
| Q2(b)    | 6 optimisers | 6 |
| Q2(c)    | Batch size × LR × epochs grid | 12 |
| **Total**|  | **24** |

All 24 runs are automatically tracked on **W&B** with per-epoch metrics, best-val-acc summaries, and a Parallel Coordinates sweep view.

---

## 📁 Repository Structure

```
vgg6-cifar10-assignment/
│
├── train.py                      ← universal CLI entry-point (all experiments)
├── evaluate.py                   ← load .pth checkpoint → report test accuracy
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── model.py                  ← VGG6 (activation injected at init)
│   ├── dataset.py                ← CIFAR-10 pipeline + augmentation
│   ├── engine.py                 ← train_one_epoch / evaluate helpers
│   ├── optimizer_factory.py      ← build any of 6 optimisers by name
│   └── utils.py                  ← set_seed, checkpointing, get_device
│
├── configs/
│   ├── __init__.py
│   └── base_config.py            ← single source of truth for defaults
│
├── scripts/
│   ├── run_q1_baseline.sh        ← 1 run  — ReLU + Adam baseline
│   ├── run_q2a_activations.sh    ← 5 runs — activation sweep
│   ├── run_q2b_optimizers.sh     ← 6 runs — optimiser sweep
│   └── run_q2c_hparams.sh        ← 12 runs — batch / LR / epoch grid
│
├── notebooks/
│   └── CS6886W_A1.ipynb          ← Colab-ready notebook (all Q1–Q3)
│
├── checkpoints/                  ← best .pth saved here (git-ignored)
└── results/                      ← optional artefact dumps
```

---

## 🧠 VGG6 Architecture

```
Input (3 × 32 × 32)
│
├─ Block 1
│   ├─ Conv2d(3 → 64, 3×3, pad=1) → BN → ACT
│   ├─ Conv2d(64 → 64, 3×3, pad=1) → BN → ACT
│   └─ MaxPool2d(2×2)           [32 → 16]
│
├─ Block 2
│   ├─ Conv2d(64 → 128, 3×3, pad=1) → BN → ACT
│   ├─ Conv2d(128 → 128, 3×3, pad=1) → BN → ACT
│   └─ MaxPool2d(2×2)          [16 → 8]
│
└─ Classifier
    ├─ Flatten → Linear(128×8×8 → 512) → ACT → Dropout(0.5)
    ├─ Linear(512 → 512) → ACT → Dropout(0.5)
    └─ Linear(512 → 10)
```

`ACT` is injected at construction time — the same `VGG6` class is used for all five activation experiments.

**Supported activations:** `relu` · `sigmoid` · `tanh` · `silu` · `gelu`

---

## 💻 Installation — Local

### 1. Clone the repo

```bash
git clone https://github.com/<YOUR_USERNAME>/vgg6-cifar10-assignment.git
cd vgg6-cifar10-assignment
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins:

```
torch>=2.1.0
torchvision>=0.16.0
wandb>=0.16.0
tqdm>=4.66.0
numpy>=1.24.0
```

### 4. Set your W&B API key

```bash
export WANDB_API_KEY="your_key_here"   # get it from https://wandb.ai/authorize
wandb login                            # or run this interactively
```

---

## 🚀 Running on Google Colab

> **All experiments can be run entirely in Google Colab — no local GPU required.**  
> The companion notebook at `notebooks/CS6886W_A1.ipynb` drives every step.

---

### Step 1 — Create a W&B Account & get your API key

1. Go to [https://wandb.ai/](https://wandb.ai/) and sign up (free).
2. Navigate to **[https://wandb.ai/authorize](https://wandb.ai/authorize)**.
3. Copy your personal API key (a 40-character string like `abc123...`).
4. Keep this tab open — you will paste the key in **Step 6**.

---

### Step 2 — Store the key as a Colab Secret

Using **Colab Secrets** means your key is never printed or stored in the notebook:

1. In Colab, click the 🔑 **key icon** in the left sidebar → "Secrets".
2. Click **+ Add new secret**.
3. Set **Name** = `WANDB_API_KEY` and **Value** = your 40-character key.
4. Toggle **"Notebook access"** ON.

---

### Step 3 — Open the Notebook

**Option A — Open directly from GitHub (recommended)**

Click the badge at the top of this README:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR_USERNAME>/vgg6-cifar10-assignment/blob/main/notebooks/CS6886W_A1.ipynb)

**Option B — Upload manually**

1. Download `notebooks/CS6886W_A1.ipynb` from this repo.
2. Go to [https://colab.research.google.com/](https://colab.research.google.com/).
3. **File → Upload notebook** → select the downloaded `.ipynb`.

---

### Step 4 — Enable GPU Runtime

> ⚠️ Always do this **before** running any cells — you cannot switch runtime mid-session.

1. In Colab: **Runtime → Change runtime type**.
2. Set **Hardware accelerator** = `T4 GPU` (free) or `A100` (Colab Pro).
3. Click **Save**.

Verify GPU is active by running:

```python
import torch
print(torch.cuda.is_available())          # True
print(torch.cuda.get_device_name(0))      # NVIDIA Tesla T4  (or similar)
```

---

### Step 5 — Mount Google Drive (optional but recommended)

Mounting Drive means your checkpoints survive Colab disconnections:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_DIR = '/content/drive/MyDrive/CS6886W_A1'
os.makedirs(DRIVE_DIR, exist_ok=True)

# Override checkpoint dir so all .pth files go to Drive
CHECKPOINT_DIR = f'{DRIVE_DIR}/checkpoints'
```

---

### Step 6 — Clone Repo & Install Dependencies

Run this cell at the top of the notebook (Cell 1):

```python
# Clone repo
!git clone https://github.com/<YOUR_USERNAME>/vgg6-cifar10-assignment.git
%cd vgg6-cifar10-assignment

# Install dependencies
!pip install -q torch torchvision wandb tqdm numpy

# Verify
import torch, wandb
print(f"PyTorch : {torch.__version__}")
print(f"W&B     : {wandb.__version__}")
print(f"GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
```

---

### Step 7 — Login to W&B using Colab Secrets

Run this cell (Cell 2 — do **not** paste the key directly into code):

```python
import os
from google.colab import userdata

# Read from Colab Secrets — key is never printed
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")

import wandb
wandb.login(key=os.environ["WANDB_API_KEY"])
print("W&B login successful ✓")
```

> **Fallback** (if Secrets are unavailable): replace with  
> `os.environ["WANDB_API_KEY"] = "paste_your_key_here"` — but **do not commit this to git**.

Set your W&B identity once (used by all subsequent runs):

```python
WANDB_ENTITY  = "<your_wandb_username>"      # ← replace with yours
WANDB_PROJECT = "CS6886W-A1-VGG6-CIFAR10"
```

---

### Step 8 — Run All Experiments

#### Q1 — Baseline (≈ 20 min on T4)

```python
!python train.py \
    --activation   relu  \
    --optimizer    adam  \
    --lr           1e-3  \
    --batch_size   128   \
    --num_epochs   30    \
    --seed         42    \
    --wandb_entity {WANDB_ENTITY} \
    --wandb_project {WANDB_PROJECT} \
    --run_name     "Q1_baseline_relu_adam"
```

Expected final output:
```
Epoch [030/030]  Train loss: 0.3412  acc: 88.21%  |  Val loss: 0.4891  acc: 84.73%
Training complete.  Best val acc = 85.12%
```

---

#### Q2(a) — Activation sweep (≈ 1.5 hrs on T4)

```python
activations = ["relu", "sigmoid", "tanh", "silu", "gelu"]

for act in activations:
    print(f"\n{'='*50}\nQ2a — activation = {act}\n{'='*50}")
    !python train.py \
        --activation   {act}  \
        --optimizer    adam   \
        --lr           1e-3   \
        --batch_size   128    \
        --num_epochs   30     \
        --seed         42     \
        --wandb_entity {WANDB_ENTITY} \
        --wandb_project {WANDB_PROJECT} \
        --run_name     "Q2a_act={act}"
```

---

#### Q2(b) — Optimiser sweep (≈ 2 hrs on T4)

```python
optimizers = {
    "sgd":          0.01,
    "nesterov_sgd": 0.01,
    "adam":         1e-3,
    "adagrad":      1e-2,
    "rmsprop":      1e-3,
    "nadam":        1e-3,
}

for opt, lr in optimizers.items():
    print(f"\n{'='*50}\nQ2b — optimizer = {opt}  lr = {lr}\n{'='*50}")
    !python train.py \
        --activation   relu   \
        --optimizer    {opt}  \
        --lr           {lr}   \
        --batch_size   128    \
        --num_epochs   30     \
        --seed         42     \
        --wandb_entity {WANDB_ENTITY} \
        --wandb_project {WANDB_PROJECT} \
        --run_name     "Q2b_opt={opt}"
```

---

#### Q2(c) — Hyperparameter grid (≈ 4 hrs on T4)

```python
import subprocess

# Batch size sweep
for bs in [32, 64, 128, 256]:
    !python train.py --activation relu --optimizer adam \
        --lr 1e-3 --batch_size {bs} --num_epochs 30 \
        --seed 42 --wandb_entity {WANDB_ENTITY} \
        --wandb_project {WANDB_PROJECT} --run_name "Q2c_bs={bs}"

# Learning rate sweep
for lr in ["1e-4", "5e-4", "1e-3", "5e-3"]:
    !python train.py --activation relu --optimizer adam \
        --lr {lr} --batch_size 128 --num_epochs 30 \
        --seed 42 --wandb_entity {WANDB_ENTITY} \
        --wandb_project {WANDB_PROJECT} --run_name "Q2c_lr={lr}"

# Epoch sweep
for ep in [10, 20, 30, 50]:
    !python train.py --activation relu --optimizer adam \
        --lr 1e-3 --batch_size 128 --num_epochs {ep} \
        --seed 42 --wandb_entity {WANDB_ENTITY} \
        --wandb_project {WANDB_PROJECT} --run_name "Q2c_ep={ep}"
```

---

#### Q3 — Fetch results & generate plots

```python
import wandb
import pandas as pd
import plotly.express as px

api = wandb.Api()
runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")

# Build summary table
records = []
for r in runs:
    records.append({
        "name":         r.name,
        "activation":   r.config.get("activation", "-"),
        "optimizer":    r.config.get("optimizer", "-"),
        "batch_size":   r.config.get("batch_size", "-"),
        "lr":           r.config.get("lr", "-"),
        "num_epochs":   r.config.get("num_epochs", "-"),
        "best_val_acc": r.summary.get("best_val_acc", None),
    })

df = pd.DataFrame(records).sort_values("best_val_acc", ascending=False)
print(df.to_string(index=False))

# ── Plot 1: Q2a — Val acc per activation ───────────────────────────────────
q2a = df[df["name"].str.startswith("Q2a")]
fig = px.bar(q2a, x="activation", y="best_val_acc",
             title="Q2(a) Best Val Acc vs Activation Function",
             labels={"best_val_acc": "Best Val Acc (%)"}, color="activation")
fig.show()

# ── Plot 2: Q2b — Val acc per optimiser ────────────────────────────────────
q2b = df[df["name"].str.startswith("Q2b")]
fig2 = px.bar(q2b, x="optimizer", y="best_val_acc",
              title="Q2(b) Best Val Acc vs Optimiser",
              labels={"best_val_acc": "Best Val Acc (%)"}, color="optimizer")
fig2.show()

# ── Plot 3: Q2c batch size ──────────────────────────────────────────────────
q2c_bs = df[df["name"].str.contains("Q2c_bs")]
fig3 = px.line(q2c_bs, x="batch_size", y="best_val_acc",
               markers=True, title="Q2(c) Val Acc vs Batch Size")
fig3.show()

# ── Plot 4: Q2c learning rate ───────────────────────────────────────────────
q2c_lr = df[df["name"].str.contains("Q2c_lr")].sort_values("lr")
fig4 = px.line(q2c_lr, x="lr", y="best_val_acc",
               markers=True, log_x=True, title="Q2(c) Val Acc vs Learning Rate")
fig4.show()

# ── Plot 5: Q2c epochs ──────────────────────────────────────────────────────
q2c_ep = df[df["name"].str.contains("Q2c_ep")].sort_values("num_epochs")
fig5 = px.line(q2c_ep, x="num_epochs", y="best_val_acc",
               markers=True, title="Q2(c) Val Acc vs Number of Epochs")
fig5.show()
```

---

## 📊 W&B Dashboard & Plots

### Accessing your project dashboard

After completing all runs:

```
https://wandb.ai/<YOUR_WANDB_USERNAME>/CS6886W-A1-VGG6-CIFAR10
```

---

### Building the required plots in W&B

#### Line chart — Training & Validation Loss/Accuracy

1. Go to your project → **Workspace** tab.
2. Click **+ Add panel** → **Line chart**.
3. Y-axis: `train/loss` or `val/acc` · X-axis: `epoch`.
4. To overlay multiple runs: use **Run Sets** → filter by run name prefix (e.g. `Q2a_`).
5. Click **Apply** to save the panel.

#### Scatter plot — Best Val Acc across runs

1. **+ Add panel** → **Scatter plot**.
2. X-axis: `best_val_acc` · Y-axis: `best_val_acc` (or any config key).
3. Color by: `optimizer` or `activation`.

#### Parallel Coordinates plot (Q3 sweep view)

1. **+ Add panel** → **Parallel Coordinates**.
2. Add axes in this order:
   - `activation`
   - `optimizer`
   - `batch_size`
   - `lr`
   - `num_epochs`
   - `best_val_acc` ← the target column (colour by this)
3. Click **Apply**.
4. Drag sliders on each axis to filter runs interactively.

#### Exporting a plot as PNG

1. Hover over any panel → click **⋮ (three dots)** → **Download → PNG / SVG**.

#### Sharing a W&B Report

1. Click **Reports** → **+ New report**.
2. Drag panels from your workspace into the report.
3. Click **Publish** (top-right) → copy the public link.

---

## 📈 Results

> ✏️ *Fill in after running all experiments. Placeholder values shown below.*

### Q2(a) — Activation Functions

| Activation | Best Val Acc (%) |
|------------|-----------------|
| ReLU       | —               |
| Sigmoid    | —               |
| Tanh       | —               |
| SiLU       | —               |
| GELU       | —               |

### Q2(b) — Optimisers

| Optimiser     | LR     | Best Val Acc (%) |
|---------------|--------|-----------------|
| SGD           | 0.01   | —               |
| Nesterov SGD  | 0.01   | —               |
| Adam          | 1e-3   | —               |
| Adagrad       | 1e-2   | —               |
| RMSprop       | 1e-3   | —               |
| NAdam         | 1e-3   | —               |

### Q2(c) — Hyperparameters

**Batch size** (Adam, lr=1e-3, 30 epochs)

| Batch Size | Best Val Acc (%) |
|------------|-----------------|
| 32         | —               |
| 64         | —               |
| 128        | —               |
| 256        | —               |

**Learning rate** (Adam, bs=128, 30 epochs)

| LR    | Best Val Acc (%) |
|-------|-----------------|
| 1e-4  | —               |
| 5e-4  | —               |
| 1e-3  | —               |
| 5e-3  | —               |

**Epochs** (Adam, lr=1e-3, bs=128)

| Epochs | Best Val Acc (%) |
|--------|-----------------|
| 10     | —               |
| 20     | —               |
| 30     | —               |
| 50     | —               |

---

## ⚙️ Project Configuration

All defaults live in `configs/base_config.py` — override any key via CLI flag:

| Parameter        | Default                    | CLI flag            |
|------------------|---------------------------|---------------------|
| `activation`     | `relu`                    | `--activation`      |
| `optimizer`      | `adam`                    | `--optimizer`       |
| `lr`             | `1e-3`                    | `--lr`              |
| `weight_decay`   | `5e-4`                    | `--weight_decay`    |
| `batch_size`     | `128`                     | `--batch_size`      |
| `num_epochs`     | `30`                      | `--num_epochs`      |
| `seed`           | `42`                      | `--seed`            |
| `data_dir`       | `./data`                  | `--data_dir`        |
| `checkpoint_dir` | `./checkpoints`           | `--checkpoint_dir`  |
| `wandb_project`  | `CS6886W-A1-VGG6-CIFAR10` | `--wandb_project`   |
| `wandb_entity`   | *(wandb default)*         | `--wandb_entity`    |

### Evaluate a saved checkpoint

```bash
python evaluate.py --checkpoint checkpoints/Q1_baseline_relu_adam_best.pth
```

Expected output:
```
Loaded checkpoint from: checkpoints/Q1_baseline_relu_adam_best.pth
  epoch = 27, saved val_acc = 85.12%

Test top-1 accuracy: 85.12%
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-change`
3. Commit: `git commit -m "feat: describe your change"`
4. Push: `git push origin feature/my-change`
5. Open a Pull Request

---

<p align="center">
  Made with ❤️ for CS6886W · Deep Learning Systems
</p>
