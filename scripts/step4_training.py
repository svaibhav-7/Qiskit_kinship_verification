"""
=============================================================
QUANTUM KINSHIP PIPELINE — STEP 4: TRAINING
=============================================================
We train the VQC parameters using COBYLA (gradient-free optimizer)
with Qiskit's Statevector simulator for exact expectation values.

Pipeline position:
    Dataset (kin/non-kin pairs) → VQC(θ) → Loss → Optimizer → Update θ
"""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ── Configuration ─────────────────────────────────────────────────────────────
N_QUBITS      = 4
N_LAYERS      = 3
TOTAL_WIRES   = 2 * N_QUBITS
EMBEDDING_DIM = 128
N_PARAMS      = N_LAYERS * TOTAL_WIRES * 3


# ── Build parameterized circuit once ──────────────────────────────────────────

def _build_circuit(n_layers=N_LAYERS):
    n_weight = n_layers * TOTAL_WIRES * 3
    input_params  = ParameterVector('x', 2 * N_QUBITS)
    weight_params = ParameterVector('w', n_weight)
    qc = QuantumCircuit(TOTAL_WIRES)
    for i in range(N_QUBITS):
        qc.ry(input_params[i], i)
    for i in range(N_QUBITS):
        qc.ry(input_params[N_QUBITS + i], N_QUBITS + i)
    for layer in range(n_layers):
        base = layer * TOTAL_WIRES * 3
        for wire in range(TOTAL_WIRES):
            idx = base + wire * 3
            qc.rz(weight_params[idx], wire)
            qc.ry(weight_params[idx+1], wire)
            qc.rz(weight_params[idx+2], wire)
        for wire in range(TOTAL_WIRES):
            qc.cx(wire, (wire + 1) % TOTAL_WIRES)
        for i in range(N_QUBITS):
            qc.cx(i, i + N_QUBITS)
    return qc, input_params, weight_params

_QC, _INP, _WGT = _build_circuit()
_Z_OP = SparsePauliOp('I' * (TOTAL_WIRES - 1) + 'Z')


def _evaluate(f1, f2, weights):
    vals = np.concatenate([f1, f2])
    d = {}
    for i, p in enumerate(_INP):
        d[p] = float(vals[i])
    for i, p in enumerate(_WGT):
        d[p] = float(weights[i])
    sv = Statevector.from_instruction(_QC.assign_parameters(d))
    return float(sv.expectation_value(_Z_OP).real)


# ── Data generation ───────────────────────────────────────────────────────────

def _base_emb(seed):
    r = np.random.default_rng(seed)
    v = r.standard_normal(EMBEDDING_DIM)
    return v / np.linalg.norm(v)

def _kin_emb(base, noise, seed):
    r = np.random.default_rng(seed)
    v = base + noise * r.standard_normal(EMBEDDING_DIM)
    return v / np.linalg.norm(v)

def _compress(emb):
    chunk = EMBEDDING_DIM // N_QUBITS
    o = np.array([np.mean(emb[i*chunk:(i+1)*chunk]) for i in range(N_QUBITS)])
    return np.pi * (o - o.min()) / (o.max() - o.min() + 1e-8)

def generate_dataset(n_kin=15, n_nonkin=15, noise=0.12, offset=0):
    X1, X2, Y = [], [], []
    for i in range(n_kin):
        base = _base_emb(offset + i)
        a = _kin_emb(base, noise, (offset+i)*100)
        b = _kin_emb(base, noise, (offset+i)*100+1)
        X1.append(_compress(a)); X2.append(_compress(b)); Y.append(+1.0)
    for i in range(n_nonkin):
        a = _base_emb(offset + 1000 + i)
        b = _base_emb(offset + 2000 + i)
        X1.append(_compress(a)); X2.append(_compress(b)); Y.append(-1.0)
    return np.array(X1), np.array(X2), np.array(Y)


# ── Loss + Training ──────────────────────────────────────────────────────────

def loss_fn(w, X1, X2, Y):
    preds = np.array([_evaluate(x1, x2, w) for x1, x2 in zip(X1, X2)])
    return float(np.mean((preds - Y)**2))

_iter = 0
_hist = []

def train(w0, X1, X2, Y, maxiter=80):
    global _iter, _hist
    _iter = 0; _hist = []
    n = len(Y)
    print(f"Training on {n} pairs for up to {maxiter} iterations...")
    print(f"  Optimizer : COBYLA | Params: {len(w0)}\n")

    def obj(w):
        global _iter, _hist
        _iter += 1
        l = loss_fn(w, X1, X2, Y)
        _hist.append(l)
        if _iter % 10 == 0 or _iter == 1:
            preds = np.array([_evaluate(x1, x2, w) for x1, x2 in zip(X1, X2)])
            acc = np.mean(np.where(preds >= 0, +1, -1) == Y) * 100
            print(f"  Iter {_iter:4d} | Loss: {l:.4f} | Acc: {acc:.1f}%")
        return l

    res = minimize(obj, w0, method='COBYLA', options={'maxiter': maxiter, 'rhobeg': 0.5})
    print(f"\n  Final loss: {res.fun:.4f}")
    return res.x, _hist


def evaluate(w, X1, X2, Y, name="Test"):
    raw = np.array([_evaluate(x1, x2, w) for x1, x2 in zip(X1, X2)])
    pred = np.where(raw >= 0, +1, -1)
    acc = np.mean(pred == Y) * 100
    tp = np.sum((pred == +1) & (Y == +1))
    fp = np.sum((pred == +1) & (Y == -1))
    fn = np.sum((pred == -1) & (Y == +1))
    prec = tp / (tp + fp + 1e-9) * 100
    rec  = tp / (tp + fn + 1e-9) * 100
    print(f"\n{'─'*45}\n{name} Results:")
    print(f"  Accuracy : {acc:.1f}% | Precision: {prec:.1f}% | Recall: {rec:.1f}%")
    print(f"{'─'*45}")
    return acc


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("STEP 4: Training the Quantum Kinship Model (Qiskit)")
    print("="*55)

    X1_tr, X2_tr, Y_tr = generate_dataset(15, 15, 0.12, 0)
    X1_te, X2_te, Y_te = generate_dataset(8, 8, 0.12, 500)
    print(f"\nTrain: {len(Y_tr)} pairs | Test: {len(Y_te)} pairs")

    rng = np.random.default_rng(42)
    w0 = rng.uniform(-np.pi/4, np.pi/4, size=N_PARAMS)

    print("\n--- Before Training ---")
    evaluate(w0, X1_tr, X2_tr, Y_tr, "Train (random)")

    print("\n--- Training ---")
    w_trained, hist = train(w0, X1_tr, X2_tr, Y_tr, maxiter=80)

    evaluate(w_trained, X1_tr, X2_tr, Y_tr, "Train (trained)")
    evaluate(w_trained, X1_te, X2_te, Y_te, "Test  (trained)")

    try:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(hist)+1), hist, lw=2, color='steelblue')
        plt.xlabel("Iteration"); plt.ylabel("MSE Loss")
        plt.title("Quantum Kinship VQC — Training Loss (Qiskit)")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig("training_loss.png", dpi=120)
        print("\n📊 Plot saved to training_loss.png")
    except Exception as e:
        print(f"Plot error: {e}")

    np.save("trained_params.npy", w_trained)
    print("💾 Params saved to trained_params.npy")
    print("\n✅ Training complete! Move on to step5_full_pipeline.py")
