# Quantum Kinship Verification Pipeline

A step-by-step implementation of quantum-based kinship detection using PennyLane.

## Pipeline Architecture

```
Image1  →  CNN  →  Embedding z1  →  Quantum Encoding  →  |ψ_z1⟩ ──┐
                                                                      ├── VQC U(θ) ──► Measure ──► ŷ ∈ {0,1}
Image2  →  CNN  →  Embedding z2  →  Quantum Encoding  →  |ψ_z2⟩ ──┘
```

## Files — Run In Order

| File | What it teaches |
|------|----------------|
| `step1_setup.py` | Install check, Hello Quantum World |
| `step2_quantum_encoding.py` | Converting face embeddings → qubit states |
| `step3_vqc.py` | Variational Quantum Circuit + SWAP test |
| `step4_training.py` | Training loop with Parameter Shift Rule |
| `step5_full_pipeline.py` | Complete end-to-end pipeline class |

## Installation

```bash
pip install pennylane pennylane-torch torch torchvision numpy matplotlib
```

## Quick Start

```bash
python step1_setup.py      # verify installation
python step2_quantum_encoding.py
python step3_vqc.py
python step4_training.py
python step5_full_pipeline.py
```

## Key Quantum Concepts Used

- **Angle Encoding** — RY(θ) maps each feature to a qubit rotation
- **Amplitude Encoding** — store full vector in qubit amplitudes  
- **Variational Quantum Circuit** — trainable RZ-RY-RZ + CNOT layers
- **Parameter Shift Rule** — exact quantum gradients (no approximation)
- **SWAP Test** — measures fidelity |⟨ψ₁|ψ₂⟩|² between two face states
- **Entanglement** — CNOT gates create correlations between face qubits

## Configuration

Edit these constants at the top of each file:

```python
N_QUBITS      = 4     # qubits per person (increase for more expressive power)
N_LAYERS      = 3     # VQC depth
EMBEDDING_DIM = 128   # CNN output size (128 for FaceNet, 512 for ArcFace)
```

## Replacing the Mock CNN

In `step5_full_pipeline.py`, find `MockCNN` and replace with your real model:

```python
# Using deepface / FaceNet
from deepface import DeepFace

def extract_embedding(image_path):
    result = DeepFace.represent(image_path, model_name="Facenet")
    emb = np.array(result[0]["embedding"])
    return emb / np.linalg.norm(emb)
```

## Real Datasets

- **KinFaceW-I / KinFaceW-II** — parent-child pairs, widely used
- **FIW (Families In the Wild)** — 1,000 families, 13 relationship types
- **Cornell KinFace** — 150 pairs, four relationship types
