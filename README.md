# Quantum Kinship Verification

A quantum-classical hybrid pipeline for kinship verification using Qiskit and PyTorch. This project uses Variational Quantum Circuits (VQC) to compare face embeddings extracted from pretrained classical CNNs.

## Project Structure

- `src/`: Core logic and modules
  - `quantum_core.py`: VQC circuit definitions and quantum utilities.
  - `data_loaders.py`: Utilities for loading KinFaceW and TSKinFace datasets.
  - `models.py`: Model classes and feature extractors (ResNet, MockCNN).
- `scripts/`: Step-by-step learning pipeline and training scripts.
  - `step1_setup.py`: Environment check and Bell state.
  - `step2_quantum_encoding.py`: Angle and Amplitude encoding demo.
  - `step3_vqc.py`: VQC and SWAP test implementation.
  - `step4_training.py`: Synthetic data training demo.
  - `step5_full_pipeline.py`: Full pipeline with mock data.
  - `train_real_data.py`: Training script for real datasets (KinFaceW-II, TSKinFace).
- `assets/`: Architecture diagrams and images.
- `weights/`: Saved model parameters (.npy).
- `results/`: Training logs and loss plots.
- `requirements.txt`: Python dependencies.
- `main.py`: Entry point for the structured pipeline.

## Getting Started

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the environment check:**
    ```bash
    python scripts/step1_setup.py
    ```

3.  **Train on real data:**
    ```bash
    python scripts/train_real_data.py
    ```

## Performance
Initial tests on synthetic data achieved ~75% accuracy. Training on real datasets (KinFaceW, TSKinFace) is currently underway to achieve higher verification performance.

## Authors
- Antigravity AI
- [Your Team Name]
