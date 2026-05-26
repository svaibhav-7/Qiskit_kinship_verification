# Quantum Kinship Verification

A high-performance quantum-classical hybrid pipeline for kinship verification using Qiskit and PyTorch. This project projects classical facial embeddings (extracted via FaceNet InceptionResnetV1) into a lower-dimensional quantum state register, and compares them using a quantum **SWAP Test** circuit to calculate kinship overlap (fidelity).

The network is conditioned on the relationship category (Father-Daughter, Father-Son, Mother-Daughter, Mother-Son) using a **Relation-Conditioned Projection** MLP, reaching **~66.89%** verification accuracy on KinFaceW-I.

---

## Project Structure

- **`src/`**: Core library modules.
  - `quantum_core.py`: Controlled-SWAP quantum circuit definitions and Qiskit C++ `AerSimulator` batch execution logic.
  - `data_loaders.py`: Parses KinFaceW-I, KinFaceW-II, and TSKinFace datasets and handles 4-category one-hot relation mapping.
  - `models.py`: Classical FaceNet feature extraction, classical projection network, and fast analytical quantum fidelity calculation.
- **`scripts/`**: Training and validation pipelines.
  - `train_hybrid.py`: Fast PyTorch analytical training with validation-based checkpointing (early stopping) and final Qiskit simulated verification.
- **`weights/`**: Directory for model checkpoints and caches.
  - `hybrid_kinship.pt`: Best-performing trained projection model weights checkpoint.
- **`results/`**: Training metrics plots (Loss & Accuracy), ROC-AUC curves, and score distributions.
- **`main.py`**: Clean interactive demonstration entry point.

---

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training & Validation
To train the relation-conditioned model (using fast analytical simulation and validating on KinFaceW-I, followed by physical Qiskit Aer simulation):
```bash
python scripts/train_hybrid.py --n-qubits 8 --epochs 40
```

### 3. Run Interactive Demo
To run the kinship prediction demo on a test face pair:
```bash
python main.py
```

---

## Verification & Metrics

The model achieves a **66.89%** validation accuracy on the test set of KinFaceW-I. The physical quantum controlled-SWAP circuit simulated using Qiskit's `AerSimulator` (1024 shots) closely matches the analytical statevector overlap:

* **Analytical Quantum Test Accuracy**: **66.89%**
* **Qiskit AerSimulator Test Accuracy**: **66.42%**
* **Precision / Recall (TPR) / F1-Score**: **65.43% / 69.61% / 67.45%**

Training curves, ROC-AUC, and overlap probability distributions are saved in the `results/` folder.
