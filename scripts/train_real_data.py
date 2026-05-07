"""
=============================================================
QUANTUM KINSHIP — REAL DATA TRAINING SCRIPT
=============================================================
Train on: KinFaceW-II + TSKinFace (real face images)
Test on : KinFaceW-I

Uses pretrained ResNet-18 for feature extraction.
"""

import os, sys, glob
import numpy as np
import scipy.io
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
KFW1 = os.path.join(BASE, "KinFaceW-I", "KinFaceW-I")
KFW2 = os.path.join(BASE, "KinFaceW-II")
TSKIN = os.path.join(BASE, "TSKinFace_Data", "TSKinFace_Data", "TSKinFace_cropped")

# ── Quantum Config ────────────────────────────────────────────────────────────
N_QUBITS    = 4
N_LAYERS    = 3
TOTAL_WIRES = 2 * N_QUBITS
N_PARAMS    = N_LAYERS * TOTAL_WIRES * 3
EMB_DIM     = 512   # ResNet-18 output dim

# ═══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE EXTRACTOR — Pretrained ResNet-18
# ═══════════════════════════════════════════════════════════════════════════════

class FaceFeatureExtractor:
    def __init__(self):
        print("  Loading pretrained ResNet-18...")
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Identity()  # Remove classification head → 512-dim
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, img_path):
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0)
        emb = self.model(tensor).squeeze().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)

    @torch.no_grad()
    def extract_batch(self, paths):
        tensors = []
        for p in paths:
            img = Image.open(p).convert('RGB')
            tensors.append(self.transform(img))
        batch = torch.stack(tensors)
        embs = self.model(batch).numpy()
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        return embs / norms


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def compress(emb, n_qubits=N_QUBITS):
    chunk = len(emb) // n_qubits
    o = np.array([np.mean(emb[i*chunk:(i+1)*chunk]) for i in range(n_qubits)])
    lo, hi = o.min(), o.max()
    return np.pi * (o - lo) / (hi - lo + 1e-8)


def load_kinfacew(root, cnn, max_pairs=None):
    """Load KinFaceW-I or KinFaceW-II using .mat metadata."""
    relations = ['fd', 'fs', 'md', 'ms']
    rel_dirs = {'fd': 'father-dau', 'fs': 'father-son',
                'md': 'mother-dau', 'ms': 'mother-son'}
    X1, X2, Y = [], [], []
    skipped = 0

    for rel in relations:
        mat_path = os.path.join(root, "meta_data", f"{rel}_pairs.mat")
        if not os.path.exists(mat_path):
            print(f"    [SKIP] {mat_path} not found")
            continue
        data = scipy.io.loadmat(mat_path)
        pairs = data['pairs']
        img_dir = os.path.join(root, "images", rel_dirs[rel])
        count = 0
        for row in pairs:
            label = int(row[1].flat[0])    # 1=kin, 0=non-kin
            img1 = str(row[2].flat[0])
            img2 = str(row[3].flat[0])
            p1 = os.path.join(img_dir, img1)
            p2 = os.path.join(img_dir, img2)
            if not os.path.exists(p1) or not os.path.exists(p2):
                skipped += 1
                continue
            e1 = cnn.extract(p1)
            e2 = cnn.extract(p2)
            X1.append(compress(e1))
            X2.append(compress(e2))
            Y.append(+1.0 if label == 1 else -1.0)
            count += 1
            if max_pairs and count >= max_pairs:
                break
        print(f"    {rel.upper()}: loaded {count} pairs")

    if skipped:
        print(f"    (skipped {skipped} pairs with missing images)")
    return np.array(X1), np.array(X2), np.array(Y)


def load_tskinface(root, cnn, max_families=200):
    """Load TSKinFace — generate parent-child kin pairs + non-kin pairs."""
    X1, X2, Y = [], [], []
    kin_embs = []   # store for non-kin generation

    for folder in ['FMS', 'FMD']:
        fdir = os.path.join(root, folder)
        if not os.path.exists(fdir):
            continue
        child_key = 'S' if folder == 'FMS' else 'D'
        families = set()
        for f in os.listdir(fdir):
            if f.endswith('.jpg'):
                parts = f.replace('.jpg','').split('-')
                if len(parts) == 3:
                    families.add(int(parts[1]))
        families = sorted(families)[:max_families]
        count = 0
        for fid in families:
            f_path = os.path.join(fdir, f"{folder}-{fid}-F.jpg")
            m_path = os.path.join(fdir, f"{folder}-{fid}-M.jpg")
            c_path = os.path.join(fdir, f"{folder}-{fid}-{child_key}.jpg")
            if not all(os.path.exists(p) for p in [f_path, m_path, c_path]):
                continue
            ef = cnn.extract(f_path)
            em = cnn.extract(m_path)
            ec = cnn.extract(c_path)
            # Father-Child kin pair
            X1.append(compress(ef)); X2.append(compress(ec)); Y.append(+1.0)
            # Mother-Child kin pair
            X1.append(compress(em)); X2.append(compress(ec)); Y.append(+1.0)
            kin_embs.append(ef); kin_embs.append(em); kin_embs.append(ec)
            count += 1
        print(f"    {folder}: {count} families -> {count*2} kin pairs")

    # Generate non-kin pairs by cross-matching
    n_kin = sum(1 for y in Y if y == 1.0)
    rng = np.random.default_rng(42)
    n_embs = len(kin_embs)
    added = 0
    while added < n_kin and n_embs > 1:
        i, j = rng.choice(n_embs, size=2, replace=False)
        X1.append(compress(kin_embs[i]))
        X2.append(compress(kin_embs[j]))
        Y.append(-1.0)
        added += 1
    print(f"    Generated {added} non-kin pairs")

    return np.array(X1), np.array(X2), np.array(Y)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. QUANTUM CIRCUIT (same VQC from step3/4)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_vqc():
    inp = ParameterVector('x', 2 * N_QUBITS)
    wgt = ParameterVector('w', N_PARAMS)
    qc  = QuantumCircuit(TOTAL_WIRES)
    for i in range(N_QUBITS):
        qc.ry(inp[i], i)
    for i in range(N_QUBITS):
        qc.ry(inp[N_QUBITS+i], N_QUBITS+i)
    for layer in range(N_LAYERS):
        base = layer * TOTAL_WIRES * 3
        for wire in range(TOTAL_WIRES):
            idx = base + wire * 3
            qc.rz(wgt[idx], wire)
            qc.ry(wgt[idx+1], wire)
            qc.rz(wgt[idx+2], wire)
        for wire in range(TOTAL_WIRES):
            qc.cx(wire, (wire+1) % TOTAL_WIRES)
        for i in range(N_QUBITS):
            qc.cx(i, i + N_QUBITS)
    return qc, inp, wgt

_QC, _INP, _WGT = _build_vqc()
_Z_OP = SparsePauliOp('I' * (TOTAL_WIRES - 1) + 'Z')

def _eval(f1, f2, w):
    vals = np.concatenate([f1, f2])
    d = {}
    for i, p in enumerate(_INP): d[p] = float(vals[i])
    for i, p in enumerate(_WGT): d[p] = float(w[i])
    sv = Statevector.from_instruction(_QC.assign_parameters(d))
    return float(sv.expectation_value(_Z_OP).real)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train(w0, X1, X2, Y, maxiter=300):
    hist = []
    counter = [0]
    n = len(Y)
    print(f"\n  Training on {n} pairs | {N_PARAMS} params | COBYLA | max {maxiter} iter")

    def obj(w):
        counter[0] += 1
        preds = np.array([_eval(x1, x2, w) for x1, x2 in zip(X1, X2)])
        loss = float(np.mean((preds - Y)**2))
        hist.append(loss)
        if counter[0] % 25 == 0 or counter[0] == 1:
            acc = np.mean(np.where(preds >= 0, +1, -1) == Y) * 100
            print(f"    Iter {counter[0]:4d} | Loss {loss:.4f} | Acc {acc:.1f}%")
        return loss

    res = minimize(obj, w0, method='COBYLA',
                   options={'maxiter': maxiter, 'rhobeg': 0.3})
    print(f"    Final loss: {res.fun:.4f}")
    return res.x, hist


def evaluate(w, X1, X2, Y, name="Test"):
    preds = np.array([_eval(x1, x2, w) for x1, x2 in zip(X1, X2)])
    pred_labels = np.where(preds >= 0, +1, -1)
    acc = np.mean(pred_labels == Y) * 100
    tp = np.sum((pred_labels == +1) & (Y == +1))
    fp = np.sum((pred_labels == +1) & (Y == -1))
    fn = np.sum((pred_labels == -1) & (Y == +1))
    tn = np.sum((pred_labels == -1) & (Y == -1))
    prec = tp / (tp + fp + 1e-9) * 100
    rec  = tp / (tp + fn + 1e-9) * 100
    print(f"\n  {name}:")
    print(f"    Accuracy  : {acc:.1f}%")
    print(f"    Precision : {prec:.1f}%")
    print(f"    Recall    : {rec:.1f}%")
    print(f"    TP={tp} FP={fp} FN={fn} TN={tn}")
    return acc


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("QUANTUM KINSHIP — Real Data Training (Qiskit + ResNet-18)")
    print("="*60)

    cnn = FaceFeatureExtractor()

    # ── Load Training Data ────────────────────────────────────────────────
    print("\n[1/4] Loading KinFaceW-II training data...")
    X1_kfw2, X2_kfw2, Y_kfw2 = load_kinfacew(KFW2, cnn)

    print("\n[2/4] Loading TSKinFace training data...")
    X1_ts, X2_ts, Y_ts = load_tskinface(TSKIN, cnn, max_families=150)

    # Combine training sets
    X1_train = np.concatenate([X1_kfw2, X1_ts])
    X2_train = np.concatenate([X2_kfw2, X2_ts])
    Y_train  = np.concatenate([Y_kfw2, Y_ts])

    n_kin = int(np.sum(Y_train == 1))
    n_nonkin = int(np.sum(Y_train == -1))
    print(f"\n  Combined training set: {len(Y_train)} pairs ({n_kin} kin, {n_nonkin} non-kin)")

    # ── Load Test Data (KinFaceW-I) ───────────────────────────────────────
    print("\n[3/4] Loading KinFaceW-I test data...")
    X1_test, X2_test, Y_test = load_kinfacew(KFW1, cnn)
    n_kin_t = int(np.sum(Y_test == 1))
    n_nonkin_t = int(np.sum(Y_test == -1))
    print(f"  Test set: {len(Y_test)} pairs ({n_kin_t} kin, {n_nonkin_t} non-kin)")

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n[4/4] Training Quantum VQC...")
    rng = np.random.default_rng(42)
    w0 = rng.uniform(-np.pi/4, np.pi/4, size=N_PARAMS)

    # Evaluate before training
    print("\n--- Before Training ---")
    evaluate(w0, X1_test, X2_test, Y_test, "KinFaceW-I (random params)")

    # Train
    print("\n--- Training on KinFaceW-II + TSKinFace ---")
    w_trained, hist = train(w0, X1_train, X2_train, Y_train, maxiter=300)

    # Evaluate after training
    print("\n--- After Training ---")
    evaluate(w_trained, X1_train, X2_train, Y_train, "Train (KFW-II + TSKin)")
    evaluate(w_trained, X1_test,  X2_test,  Y_test,  "Test  (KinFaceW-I)")

    # ── Save ──────────────────────────────────────────────────────────────
    np.save("trained_params_real.npy", w_trained)
    print("\nSaved trained_params_real.npy")

    # ── Plot ──────────────────────────────────────────────────────────────
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(hist)+1), hist, lw=2, color='steelblue')
        plt.xlabel("Iteration"); plt.ylabel("MSE Loss")
        plt.title("Quantum Kinship VQC — Real Data Training Loss")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig("training_loss_real.png", dpi=120)
        print("Saved training_loss_real.png")
    except Exception as e:
        print(f"Plot error: {e}")

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
