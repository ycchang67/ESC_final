# Composition Assistant

A deep learning system that scores the overall aesthetic quality of portrait photographs — evaluating lighting, tone, composition, and visual harmony.

Implementation in this repo now lives under `portraiq/` and is scaffolded with runnable training/inference entry points.

> **Note**
> This project is trained using AVA-based scoring, general portrait images as **Good/Acceptable**, and more professional photography as **Excellent**. Model outputs are for reference only and do not represent an absolute aesthetic judgment.

---

## Overview

Casual photographers often struggle to consistently produce visually strong portraits without formal training or curated feedback. This project addresses that gap by training a model to estimate overall portrait aesthetic quality and output a continuous score from **0.0 to 10.0**.

The scoring system combines two complementary sources:

- **AI-learned score** — trained from a large corpus of portrait images with human preference signals, learning broad aesthetic patterns

**Score scale:**

| Range | Rating |
|-------|--------|
| 0.0 – 5.0 | Poor |
| 5.0 – 7.5 | Acceptable |
| 7.5 – 9.0 | Good |
| 9.0 – 10.0 | Excellent |

These four levels represent overall aesthetic quality tiers rather than position-only composition quality.

---

## Project Structure

```
portraiq/
│
├── data/                            # Dataset (fully independent from all code)
│   ├── raw/                         # Original portrait images, untouched
│   │   ├── level1_poor/             # AVA score < 5.0
│   │   ├── level2_acceptable/       # AVA score 5.0–7.5
│   │   ├── level3_good/             # AVA score 7.5–9.0
│   │   └── level4_excellent/        # Pexels + Unsplash portraits for excellent class
│   ├── processed/                   # Training-ready portrait images (person-filtered)
│   │   ├── level1_poor/             # AVA score < 5.0
│   │   ├── level2_acceptable/       # AVA score 5.0–7.5
│   │   ├── level3_good/             # AVA score 7.5–9.0
│   │   └── level4_excellent/        # Pexels + Unsplash portraits for excellent class
│   ├── annotations/                 # Per-image score labels (includes level3_ffhq.json, level4_excellent.json, level4_smugmug.json)
│   │                                # train/val/test split is stored per record in annotation JSON
│   └── README.md                    # Data collection/filtering/annotation guide
│
├── models/                          # Model definitions & weights (shared by both pipelines)
│   ├── backbone/                    # Feature extractor (CLIP ViT-L/14)
│   ├── scorer/                      # Regression scoring head (outputs 0.0–10.0)
│   ├── best.pth                     # Accurate model checkpoint (CLIP profile)
│   ├── best_lite.pth                # Lightweight model checkpoint (EfficientNet-B4 profile)
│   └── configs/                     # YAML hyperparameter files per experiment
│
├── training/                        # ── TRAINING PIPELINE (standalone) ──
│   ├── dataset.py                   # PyTorch Dataset class definition
│   ├── train.py                     # Main training script (multi-GPU accelerated)
│   ├── evaluate.py                  # Validation & test set evaluation
│   ├── augment.py                   # Data augmentation pipeline
│   └── losses.py                    # Loss functions (Smooth L1 + optional variance penalty)
│
├── inference/                       # ── EXECUTION PIPELINE (standalone) ──
│   ├── predict.py                   # Single-image scoring entry point
│   └── visualize.py                 # Overlay score and prediction outputs
│
├── utils/                           # Shared utilities — imported by both pipelines
│   ├── image_utils.py               # Common Pillow / tensor transform helpers
│   ├── pose_utils.py                # Person detection wrapper (YOLO / MediaPipe)
│   ├── gpu_config.py                # CUDA device selection & memory management
│   └── logger.py                    # Training log & experiment tracking
│
├── main_train.py                    # ▶ TRAINING entry point — runs training pipeline only
├── main_infer.py                    # ▶ EXECUTION entry point — scores input images only
├── requirements_train.txt           # Dependencies for training (includes tensorboard, etc.)
├── requirements_infer.txt           # CPU-friendly inference dependencies
├── requirements_infer_full.txt      # Full inference dependencies (YOLO + CLIP)
├── requirements_infer_rpi4.txt      # Raspberry Pi helper deps (torch installed separately)
├── config.yaml                      # Global project configuration
├── config_train_mobilenet.yaml      # Lightweight training config (EfficientNet-B4 profile for Raspberry Pi deployment)
├── config_infer_rpi4.yaml           # Raspberry Pi 4 CPU inference config
└── README.md
```

Repository root also includes `.gitignore` to exclude datasets, model artifacts, cache files, and local runtime outputs from GitHub.

### Two Independent Entry Points

| Script | Purpose | Requires |
|--------|---------|---------|
| `main_train.py` | Train the model, output `.pth` checkpoints | `requirements_train.txt` |
| `main_infer.py` | Score new portrait images using saved weights | `requirements_infer.txt` (CPU-friendly) |

The **only connection** between the two pipelines is the trained `.pth` files under `models/` (for example `models/best.pth` and `models/best_lite.pth`). The execution environment does not need any training dependencies installed.

---

## Module Design Principles

### Separation of concerns

Training code (`training/`) and inference code (`inference/`) are **completely separated**. Both import shared logic from `models/` and `utils/` via standard Python imports — no logic is duplicated. This allows the inference pipeline to be shipped independently without pulling in training dependencies.

### Dataset independence

The `data/` directory is self-contained. Labels are stored as JSON in `annotations/` alongside image paths, making it straightforward to swap datasets or add new annotation rounds without touching any model or training code.

### GPU acceleration

All CUDA configuration is centralized in `utils/gpu_config.py`. The current workstation runs **1× NVIDIA GeForce RTX 4070 SUPER (12 GB VRAM)** with CUDA 12.9 (driver 575.64). GPU memory allocation, mixed-precision training (`torch.cuda.amp`), optional data parallelism, and device fallback to CPU are all handled in one place. With this setup, CLIP backbones are practical with moderate batch sizes and input resolutions.

---

## Model Architecture

### Backbone — Feature Extractor

| Option | Parameters | Checkpoint Size | Input Size | Notes |
|--------|------------|-----------------|------------|-------|
| **EfficientNet-B4** | **~19M** | **~221MB (current run)** | **224×224** | **Lightweight deployment profile** |
| CLIP ViT-L/14 | ~307M | ~3.48GB (current run) | 224×224 | High-accuracy workstation model |

The current codebase uses two backbone profiles: **EfficientNet-B4** (deployment/lightweight path) and **CLIP ViT-L/14** (high-accuracy path). ViT-L/14 produces richer global features for aesthetics, while EfficientNet-B4 keeps runtime and model size lower for edge deployment.

### Scoring Head — Regression

A lightweight-but-higher-capacity regression head attached to the backbone output:

```
Backbone Features → Linear(D, 512) → ReLU → Dropout(0.3)
                 → Linear(512, 256) → ReLU → Dropout(0.2)
                 → Linear(256, 64) → ReLU → Dropout(0.1)
                 → Linear(64, 1) → Clamp(0, 10)
```

`D` depends on backbone (`clip_vit_l14=768`, `efficientnet_b4=1792`) and is selected automatically in `models/backbone/factory.py`. Output is a single float in [0.0, 10.0].

For Raspberry Pi deployment, the EfficientNet-B4 profile currently uses `224×224` input resolution with checkpoint size around **221MB** in this workspace.

### Training Strategy

Current training uses:

- **Two-phase freeze/unfreeze training**:  
  Phase 1 (`epoch 1..freeze_epochs`) freezes backbone parameters and trains only the scoring head.  
  Phase 2 (`freeze_epochs+1..end`) unfreezes the backbone and continues fine-tuning with `backbone_lr_scale`.
- **Gradient accumulation** (`gradient_accumulation_steps: 4`) to simulate larger effective batch size without exceeding VRAM limits.  
  Effective batch size = `batch_size × gradient_accumulation_steps` (for example, `8 × 4 = 32`).
- **Weighted sampling** across `level1_poor`, `level2_acceptable`, `level3_good`, and `level4_excellent` to reduce class-imbalance bias from AVA-heavy levels.
- **Smooth L1 (Huber) loss** for stronger robustness to noisy/outlier score labels.
- **Variance penalty regularization** (`variance_penalty: 0.1`) to discourage collapsed predictions around the mean.
- **CosineAnnealingLR** schedule from initial LR down to `1e-6` across total epochs.
- **Early stopping** on validation MAE (`early_stopping_patience: 7`), with countdown paused during Phase 1 and active from Phase 2 onward.
- **Validation prediction-distribution logging** per epoch (`pred_score_min/mean/max/std`) to monitor range compression.

### Scoring Pipeline (Inference)

```
Input Image
    │
    ▼
Person Detection (optional utility)
    │
    └──▶ AI Model Score                ← backbone + scorer
             Learned aesthetic prediction
    │
    ▼
Final Score = AI_score
    │
    ▼
Visualize & Output (visualize.py)
```

---

## Dataset Strategy

All data collection, filtering, and annotation instructions were moved to:

- `portraiq/data/README.md`

This includes AVA two-stage collection, FFHQ/Level-4 source collection, annotation format, and data troubleshooting.

---

## Environment & Requirements

Use these two platform flows only:

### Shared Model File Placement (RTX 4070 + Raspberry Pi 4)

Place model files in the same location for both platforms:

```text
portraiq/models/best.pth        # accurate model (CLIP)
portraiq/models/best_lite.pth   # lightweight model (EfficientNet-B4)
```

Model download links:
- Accurate model: https://huggingface.co/Rax1010/Humanframe-AI
- Lightweight model: https://huggingface.co/Rax1010/Humanframe-AI-lite

### RTX 4070 (Training + Inference)

Install packages:
```bash
cd ~/Humanframe-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir numpy==1.26.4
python -m pip install --no-cache-dir \
  torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-cache-dir opencv-python==4.10.0.84
python -m pip install -r portraiq/requirements_train.txt
python -m pip install datasets
```

Training:
```bash
cd ~/Humanframe-AI/portraiq
source ../.venv/bin/activate
python main_train.py --config config.yaml
```

Inference:
```bash
cd ~/Humanframe-AI/portraiq
source ../.venv/bin/activate
python3 main_infer.py --config config.yaml --image path/to/photo.jpg --checkpoint models/best.pth
```

### Raspberry Pi 4 (Inference)

Install packages:
```bash
cd /home/pi/Humanframe-AI/portraiq
sudo apt update
sudo apt install -y python3-venv python3-full

# Rebuild a clean venv (prevents pip corruption issues such as
# "ValueError: bad marshal data (unknown type code)")
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade --force-reinstall pip setuptools wheel
python -m pip install --no-cache-dir \
  --extra-index-url https://www.piwheels.org/simple \
  torch==2.2.2 torchvision==0.17.2
python -m pip install --no-cache-dir numpy==1.26.4
python -m pip install --no-cache-dir -r requirements_infer_rpi4.txt

# If pinned torch/torchvision installation fails, fallback:
pip install torchvision
pip install torch
```

Inference:
```bash
cd ~/Humanframe-AI/portraiq
source .venv/bin/activate
python3 main_infer.py --config config_infer_rpi4.yaml --image path/to/photo.jpg --checkpoint models/best_lite.pth --no_overlay
```

Optional API-style inference on GPU:
```bash
cd ~/Humanframe-AI/portraiq
source ../.venv/bin/activate
python3 human_predict_api.py --image ./photos/test13.jpg --profile accurate
```

Optional API-style inference on CPU (Raspberry Pi):
```bash
cd /home/pi/Humanframe-AI/portraiq
source .venv/bin/activate
python3 human_predict_api.py --image ./photos/test13.jpg --profile lightweight
```

API profile defaults:
- `accurate` -> `models/best.pth`
- `lightweight` -> `models/best_lite.pth`

Dataset collection instructions are documented in:
`portraiq/data/README.md`

---

## Configuration Files

Use the provided config files directly instead of copying settings from README:

- `portraiq/config.yaml` (default training/inference on workstation)
- `portraiq/config_train_mobilenet.yaml` (EfficientNet-B4 lightweight training profile)
- `portraiq/config_infer_rpi4.yaml` (Raspberry Pi inference profile)

If you need custom hyperparameters, duplicate one of the YAML files and pass it with `--config`.

---

## License

MIT License. Dataset usage is subject to each source's individual terms of service.
