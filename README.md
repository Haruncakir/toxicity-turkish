# Team 07 – TOXICITY: Explainable Turkish Implicit Toxicity Detection

This repository contains the research development environment for **Explainable Turkish Implicit Toxicity Detection**.

---

## 1. System & Runtime Requirements

- **Operating System:** Linux x86_64 (tested on Ubuntu 24.04 / Zorin OS 18.1)
- **Python Version:** `>=3.11,<3.13` (tested with CPython 3.11.18 managed by `uv`)
- **Dependency Manager:** [uv](https://docs.astral.sh/uv/) (`>=0.5.0`, tested with `0.13.7`)
- **Hardware Acceleration (Local):** NVIDIA GPU with driver supporting CUDA 12.x/13.x (tested with NVIDIA GeForce RTX 4070 Laptop GPU, Driver 580.178.04)
- **Fallback Acceleration (Cloud):** Google Colab or Kaggle GPU for BERTurk fine-tuning if a local NVIDIA GPU is unavailable.

---

## 2. Environment Initialization

All environment setup and dependency resolutions are managed exclusively via `uv`.

### 2.1 Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Ensure `~/.local/bin` is in your `PATH`.

### 2.2 Initialize Virtual Environment
Initialize an isolated virtual environment pinned to Python 3.11:
```bash
uv venv --python 3.11
```
The interpreter configuration is stored in `.python-version`.

---

## 3. Dependency Management & Installation

The project separates dependencies into standard groups in `pyproject.toml`:
- **Core Dependencies:** `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`, `torch`, `accelerate`, `transformers`, `datasets`, `evaluate`, `huggingface-hub`, `safetensors`, `shap`, `matplotlib`, `seaborn`, `tqdm`, `pyyaml`, `psutil`.
- **Development (`dev`):** `pytest`, `pytest-cov`, `ruff`.
- **Notebooks (`notebook`):** `jupyterlab`, `ipykernel`, `ipywidgets`.
- **Data Acquisition (`data-acquisition`):** `kagglehub`.

### 3.1 Full Development Installation
To restore and install all dependencies (including dev, notebook, and data acquisition groups) into `.venv/`:
```bash
uv sync --all-groups
```

### 3.2 Core Installation Only (Production / Minimal)
```bash
uv sync --no-dev
```

---

## 4. PyTorch & GPU Configuration

### 4.1 Understanding Driver vs. PyTorch CUDA Versions
When checking `nvidia-smi`, the header displays:
```text
NVIDIA-SMI 580.178.04   Driver Version: 580.178.04   CUDA Version: 13.0
```
- **Driver CUDA Version (13.0):** Indicates the *maximum* CUDA driver API version supported by the installed display driver (580.178.04).
- **PyTorch CUDA Wheel (12.4):** PyTorch wheels bundle their own CUDA runtime libraries (e.g. cuBLAS, cuDNN, NCCL). NVIDIA drivers are strictly backward-compatible with any runtime version $\le$ the driver's supported CUDA version ($13.0 \ge 12.4$).
- Official stable PyTorch release wheels currently target CUDA 12.4 (`cu124`).

In `pyproject.toml`, the wheel source is configured explicitly to prevent dependency confusion:
```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cu124" },
]
```

### 4.2 Verifying Local GPU Detection
Run:
```bash
uv run python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
Expected output:
```text
CUDA Available: True
Device: NVIDIA GeForce RTX 4070 Laptop GPU
```

### 4.3 Running Without a Local GPU (CPU Fallback)
If developing on a machine without an NVIDIA GPU:
1. `uv` will install and run all CPU-based baselines, preprocessing, scientific computing, and unit tests seamlessly.
2. To configure CPU-only PyTorch explicitly, set the index in `pyproject.toml` to:
   ```toml
   [[tool.uv.index]]
   name = "pytorch-cpu"
   url = "https://download.pytorch.org/whl/cpu"
   explicit = true
   ```
   and re-run `uv lock && uv sync --all-groups`.

### 4.4 Cloud GPU Training (Google Colab / Kaggle)
For deep learning training runs (e.g., fine-tuning `dbmdz/bert-base-turkish-cased` or `dbmdz/bert-base-turkish-128k-uncased`) on cloud platforms:
1. **Google Colab:**
   - Change runtime to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
   - Clone the repository:
     ```bash
     !git clone <REPO_URL>
     %cd <REPO_DIR>
     !pip install uv
     !uv sync --all-groups
     ```
2. **Kaggle Notebooks:**
   - Attach accelerator (**GPU T4 x2** or **P100**).
   - Install `uv` and sync repository dependencies.

---

## 5. Verification & Testing

### 5.1 Environment Validation Report
Execute the comprehensive validation script:
```bash
uv run python scripts/check_environment.py
```
This inspects the Python runtime, package versions, PyTorch CUDA initialization, NLP configurations, and SHAP explainability imports without downloading any model weights.

### 5.2 Running the Test Suite
Execute the pytest suite:
```bash
uv run pytest
```
Covers:
- Scientific stack validation (NumPy dot product, Pandas DataFrames, SciPy statistics).
- Scikit-learn Logistic Regression baseline training and prediction.
- PyTorch tensor operations on CPU and GPU (if CUDA is detected).
- Hugging Face Transformers configuration validation without downloading external checkpoints.
- Explainability (SHAP) and visualization (Matplotlib, Seaborn) imports.

### 5.3 Lockfile Consistency Check
Verify that `uv.lock` is up-to-date with `pyproject.toml`:
```bash
uv lock --check
```

### 5.4 Code Quality / Linting
```bash
uv run ruff check .
```

---

## 6. Project Layout

```text
.
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── README.md
├── scripts/
│   └── check_environment.py     # Environment and hardware validation script
├── src/
│   └── toxicity_detection/      # Main package source
└── tests/
    └── test_environment.py     # Minimal environment verification test suite
```
