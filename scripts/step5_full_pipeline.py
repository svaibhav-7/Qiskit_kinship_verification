"""
=============================================================
QUANTUM KINSHIP PIPELINE — STEP 5: FULL END-TO-END PIPELINE
=============================================================
Complete assembled pipeline matching your diagram:

    Image1 → FeatureExtraction → Embedding z1 → QuantumEncoding → |ψ_z1⟩ ──┐
                                                                              ├─ VQC U(θ) → Measure → Decision
    Image2 → FeatureExtraction → Embedding z2 → QuantumEncoding → |ψ_z2⟩ ──┘

Uses Qiskit for all quantum operations.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_QUBITS      = 4
N_LAYERS      = 3
TOTAL_WIRES   = 2 * N_QUBITS
EMBEDDING_DIM = 128
THRESHOLD     = 0.0
N_PARAMS      = N_LAYERS * TOTAL_WIRES * 3


# ── STAGE 1: FEATURE EXTRACTION (Classical CNN placeholder) ──────────────────

class MockCNN:
    """Placeholder for a real face feature extractor (FaceNet, ArcFace)."""
    def __init__(self, embedding_dim=EMBEDDING_DIM, seed=0):
        self.embedding_dim = embedding_dim
        self._rng = np.random.default_rng(seed)

    def __call__(self, image_or_id):
        if isinstance(image_or_id, int):
            rng = np.random.default_rng(image_or_id)
            raw = rng.standard_normal(self.embedding_dim)
        else:
            raw = self._rng.standard_normal(self.embedding_dim)
        return raw / np.linalg.norm(raw)


# ── STAGE 2+3: EMBEDDING COMPRESSION → QUANTUM ENCODING ─────────────────────

class EmbeddingToQuantum:
    """Bridges classical CNN output and quantum circuit."""
    def __init__(self, embedding_dim=EMBEDDING_DIM, n_qubits=N_QUBITS):
        self.chunk_size = embedding_dim // n_qubits
        self.n_qubits = n_qubits

    def compress(self, embedding):
        out = np.array([
            np.mean(embedding[i*self.chunk_size:(i+1)*self.chunk_size])
            for i in range(self.n_qubits)
        ])
        lo, hi = out.min(), out.max()
        return np.pi * (out - lo) / (hi - lo + 1e-8)


# ── STAGE 4: VARIATIONAL QUANTUM CIRCUIT (Qiskit) ───────────────────────────

def _build_vqc():
    """Build the parameterized kinship VQC."""
    inp = ParameterVector('x', 2 * N_QUBITS)
    wgt = ParameterVector('w', N_PARAMS)
    qc  = QuantumCircuit(TOTAL_WIRES)

    # Encode both persons
    for i in range(N_QUBITS):
        qc.ry(inp[i], i)
    for i in range(N_QUBITS):
        qc.ry(inp[N_QUBITS + i], N_QUBITS + i)

    # VQC layers
    for layer in range(N_LAYERS):
        base = layer * TOTAL_WIRES * 3
        for wire in range(TOTAL_WIRES):
            idx = base + wire * 3
            qc.rz(wgt[idx], wire)
            qc.ry(wgt[idx+1], wire)
            qc.rz(wgt[idx+2], wire)
        for wire in range(TOTAL_WIRES):
            qc.cx(wire, (wire + 1) % TOTAL_WIRES)
        for i in range(N_QUBITS):
            qc.cx(i, i + N_QUBITS)

    return qc, inp, wgt

_QC, _INP, _WGT = _build_vqc()
_Z_OP = SparsePauliOp('I' * (TOTAL_WIRES - 1) + 'Z')


def _run_circuit(f1, f2, weights):
    """Evaluate ⟨Z⟩ on qubit 0 for a pair of feature vectors."""
    vals = np.concatenate([f1, f2])
    d = {}
    for i, p in enumerate(_INP):
        d[p] = float(vals[i])
    for i, p in enumerate(_WGT):
        d[p] = float(weights[i])
    sv = Statevector.from_instruction(_QC.assign_parameters(d))
    return float(sv.expectation_value(_Z_OP).real)


# ── FULL PIPELINE CLASS ──────────────────────────────────────────────────────

class QuantumKinshipModel:
    """
    End-to-end Quantum Kinship Verification Model (Qiskit).

    Usage:
        model = QuantumKinshipModel()
        model.train(X1, X2, Y)
        result = model.predict(image1, image2)
    """

    def __init__(self, n_layers=N_LAYERS, seed=42):
        self.feature_extractor = MockCNN(seed=seed)
        self.embedder = EmbeddingToQuantum()
        self.n_layers = n_layers
        rng = np.random.default_rng(seed)
        self.params = rng.uniform(-np.pi/4, np.pi/4, size=N_PARAMS)
        self.loss_history = []

    def get_quantum_state(self, image):
        embedding = self.feature_extractor(image)
        return self.embedder.compress(embedding)

    def score(self, image1, image2):
        f1 = self.get_quantum_state(image1)
        f2 = self.get_quantum_state(image2)
        return _run_circuit(f1, f2, self.params)

    def predict(self, image1, image2):
        s = self.score(image1, image2)
        pred = 1 if s > THRESHOLD else 0
        conf = abs(s) * 100
        return pred, round(s, 4), round(conf, 1)

    def _loss(self, w, X1, X2, Y):
        preds = np.array([_run_circuit(x1, x2, w) for x1, x2 in zip(X1, X2)])
        return float(np.mean((preds - Y)**2))

    def train(self, X1, X2, Y, maxiter=200):
        from scipy.optimize import minimize
        n = len(Y)
        counter = [0]

        def obj(w):
            counter[0] += 1
            l = self._loss(w, X1, X2, Y)
            self.loss_history.append(l)
            if counter[0] % 20 == 0:
                preds = np.where(
                    np.array([_run_circuit(x1, x2, w)
                              for x1, x2 in zip(X1, X2)]) >= 0, +1, -1)
                acc = np.mean(preds == Y) * 100
                print(f"  Iter {counter[0]:3d} | Loss {l:.4f} | Acc {acc:.1f}%")
            return l

        res = minimize(obj, self.params, method='COBYLA',
                       options={'maxiter': maxiter, 'rhobeg': 0.5})
        self.params = res.x

    def save(self, path="model_params.npy"):
        np.save(path, self.params)
        print(f"💾 Parameters saved to {path}")

    def load(self, path="model_params.npy"):
        self.params = np.load(path, allow_pickle=False)
        print(f"✅ Parameters loaded from {path}")


# ── Generate training data ───────────────────────────────────────────────────

def generate_training_data(model, n_families=15, noise=0.12):
    X1, X2, Y = [], [], []
    cnn = model.feature_extractor
    bridge = model.embedder
    families = [cnn(i * 100) for i in range(n_families)]

    for i, base in enumerate(families):
        rng = np.random.default_rng(i)
        a = base + noise * rng.standard_normal(EMBEDDING_DIM)
        b = base + noise * rng.standard_normal(EMBEDDING_DIM)
        a /= np.linalg.norm(a); b /= np.linalg.norm(b)
        X1.append(bridge.compress(a))
        X2.append(bridge.compress(b))
        Y.append(+1.0)

    for i in range(n_families):
        j = (i + 7) % n_families
        X1.append(bridge.compress(families[i]))
        X2.append(bridge.compress(families[j]))
        Y.append(-1.0)

    return np.array(X1), np.array(X2), np.array(Y)


# ── Main demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("STEP 5: Full Quantum Kinship Pipeline (Qiskit)")
    print("="*55)

    model = QuantumKinshipModel(n_layers=N_LAYERS, seed=42)
    print(f"\n✅ Model created")
    print(f"   CNN embedding dim : {EMBEDDING_DIM}")
    print(f"   Qubits per person : {N_QUBITS}")
    print(f"   VQC layers        : {N_LAYERS}")
    print(f"   Trainable params  : {N_PARAMS}")

    print("\n--- Before Training (random parameters) ---")
    pred, score, conf = model.predict(image1=1, image2=2)
    label = "KIN ✅" if pred == 1 else "NOT KIN ❌"
    print(f"Person 1 vs Person 2: {label} | score={score} | conf={conf}%")

    print("\n--- Generating Training Data ---")
    X1, X2, Y = generate_training_data(model, n_families=25, noise=0.10)
    print(f"Dataset: {len(Y)} pairs ({int(np.sum(Y==1))} kin, {int(np.sum(Y==-1))} non-kin)")

    print("\n--- Training ---")
    model.train(X1, X2, Y, maxiter=200)

    print("\n--- After Training ---")
    cnn = model.feature_extractor
    bridge = model.embedder
    base = cnn(999)
    rng = np.random.default_rng(999)
    kin_a = base + 0.1 * rng.standard_normal(EMBEDDING_DIM)
    kin_b = base + 0.1 * rng.standard_normal(EMBEDDING_DIM)
    kin_a /= np.linalg.norm(kin_a); kin_b /= np.linalg.norm(kin_b)
    s1 = _run_circuit(bridge.compress(kin_a), bridge.compress(kin_b), model.params)
    print(f"Sibling pair (kin)   : {'KIN ✅' if s1>0 else 'NOT KIN ❌'} | score={s1:.4f}")

    s_a = cnn(111); s_b = cnn(222)
    s2 = _run_circuit(bridge.compress(s_a), bridge.compress(s_b), model.params)
    print(f"Strangers (non-kin)  : {'KIN ✅' if s2>0 else 'NOT KIN ❌'} | score={s2:.4f}")

    all_p = np.array([_run_circuit(x1, x2, model.params) for x1, x2 in zip(X1, X2)])
    acc = np.mean(np.where(all_p >= 0, +1, -1) == Y) * 100
    print(f"\nFinal training accuracy : {acc:.1f}%")

    model.save("trained_params.npy")

    print("\n" + "="*55)
    print("🎉 FULL PIPELINE COMPLETE!")
    print("="*55)
    print("""
Next steps:
  1. Replace MockCNN with FaceNet / ArcFace (torchvision / deepface)
  2. Use a real kinship dataset (KinFaceW-I, KinFaceW-II, TSKinFace)
  3. Increase N_QUBITS and N_LAYERS for more expressive power
  4. Try TorchConnector for hybrid classical-quantum training
  5. Run on real IBM Quantum hardware via IBM Quantum Platform
""")
