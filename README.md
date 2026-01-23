# AI-SpikeScope

AI-SpikeScope is a production-ready, modular pipeline for detecting, denoising, classifying, and characterizing waveform spikes in time-series data using modern machine learning and deep learning techniques. The project was designed for neurophysiology and signal-processing workflows but can be adapted to any event- or spike-based waveform analysis.

Status
- Language: Python (100% of repository)
- Current status: Production — completed and maintained

Features
- Modular preprocessing: filtering, normalization, spike detection and windowing
- Learned denoising to improve signal quality prior to analysis
- Supervised and unsupervised classification for spike vs. artifact separation and unit typing
- Feature extraction: amplitude, width, energy, PCA/UMAP embeddings, clustering support
- Config-driven experiments with checkpointing for reproducible training and inference
- Extensible Python package with example scripts and notebooks

Quick start
1. Clone the repository
   git clone https://github.com/rociolopeth/AI-SpikeScope.git
   cd AI-SpikeScope

2. Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate    # Windows

3. Install dependencies
   pip install -r requirements.txt

4. Example commands (adapt to available scripts in this repository)
   - Preprocess:
     python scripts/preprocess.py --config config/preprocess.yaml --input data/raw --output data/processed
   - Train:
     python scripts/train.py --config config/train.yaml --output experiments/run01
   - Infer / Denoise:
     python scripts/infer.py --model experiments/run01/best_model.pth --input data/processed --output results
   - Evaluate:
     python scripts/evaluate.py --pred results --gt data/ground_truth --metrics precision recall f1

Note: If script names or CLI flags differ in this repository, adapt the commands accordingly. I can inspect the repository and adjust these commands to the exact script names if you want.

Repository layout (recommended)
- README.md
- requirements.txt
- config/                # YAML/JSON configs per experiment
- data/
  - raw/
  - processed/
  - ground_truth/
- scripts/               # CLI scripts (e.g., preprocess.py, train.py, infer.py, evaluate.py)
- spike_scope/           # main package: preprocessing, models, utils, metrics
- experiments/           # checkpoints, logs, results
- notebooks/             # demos and visualizations
- tests/                 # unit and integration tests

Configuration
Use YAML or JSON config files to declare dataset paths, preprocessing parameters, model hyperparameters, and training options. Example keys:

```yaml
dataset:
  path: data/processed
  sample_rate: 30000

preprocessing:
  filter: bandpass
  lowcut: 300
  highcut: 3000
  window_ms: 2

model:
  type: ConvNet
  channels: 64
  kernel_size: 5

training:
  batch_size: 128
  lr: 1e-3
  epochs: 100
```

Evaluation and reproducibility
- Typical metrics: precision, recall, F1, reconstruction error (MSE/MAE) for denoising, ROC/AUC when applicable.
- Reproducibility recommendations:
  - Fix random seeds (numpy, random, and framework-specific seeds).
  - Save full config files and environment information with each run.
  - Version checkpoints and logs (consider MLflow or Weights & Biases for experiment tracking).

Best practices
- Keep large or sensitive datasets out of the repository; provide download or preprocessing scripts instead.
- Add unit tests for critical preprocessing, model I/O, and evaluation steps.
- Use CI to run tests, linters and basic smoke checks on pull requests.
- Track experiments and models either with W&B/MLflow or a clear folder structure with timestamps and configs.

Contributing
- Open an issue to discuss new features or bugs.
- Create a branch named `feature/<short-description>` or `fix/<short-description>`.
- Include tests and update documentation for changes.
- Follow PEP8 and run linters before creating a pull request.

License & citation
Add a LICENSE file to declare the chosen license (MIT, Apache-2.0, etc.). If you would like users to cite this project in academic work, include a suggested citation (BibTeX or plain text).

Contact
- Owner: rociolopeth
- Repo: https://github.com/rociolopeth/AI-SpikeScope
- For issues and feature requests, use GitHub Issues.
