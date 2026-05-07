"""
=============================================================
QUANTUM KINSHIP PIPELINE — STEP 3: VARIATIONAL QUANTUM CIRCUIT
=============================================================
The VQC is the TRAINABLE part of the pipeline — like the layers
of a neural network, but made of quantum gates.

Pipeline position:
    |ψ_z1⟩ + |ψ_z2⟩ → [VQC U(θ)] → Measurement → Decision
"""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
N_QUBITS = 4      # qubits per person (so total = 8 for two people)
N_LAYERS = 3      # depth of the VQC — more layers = more expressive

# We use 2×N_QUBITS wires total:
#   Qubits 0..3  → Person 1's face state |ψ_z1⟩
#   Qubits 4..7  → Person 2's face state |ψ_z2⟩
TOTAL_WIRES = 2 * N_QUBITS


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCK 1: Encoding layer
# ─────────────────────────────────────────────────────────────────────────────
# Encodes one person's features into their qubits using RY gates.
# Called once for each person before the VQC runs.

def encode_person(qc, features, wire_offset):
    """
    Encode face features into qubits starting at wire_offset.

    Args:
        qc          : QuantumCircuit to add gates to
        features    : array of N_QUBITS angles (from the embedding)
        wire_offset : 0 for person1, N_QUBITS for person2
    """
    for i in range(N_QUBITS):
        qc.ry(features[i], wire_offset + i)


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCK 2: One VQC layer
# ─────────────────────────────────────────────────────────────────────────────
# Each layer has two parts:
#   (a) Rotation gates with trainable parameters θ  ← the "weights"
#   (b) CNOT entangling gates  ← fixed, creates quantum correlations
#
# Think of this like one layer of a neural network:
#   Linear(weights) → Activation → next layer

def vqc_layer(qc, params, layer_idx):
    """
    One layer of the Variational Quantum Circuit.

    Args:
        qc        : QuantumCircuit to add gates to
        params    : flat list/array of parameters for this layer
                    — 3 rotation angles per qubit (TOTAL_WIRES * 3 per layer)
        layer_idx : which layer we are in (for labeling)
    """
    # Part A: Parameterized rotations on every qubit
    # RZ(φ) → RY(θ) → RZ(ω) — a universal single-qubit rotation
    for wire in range(TOTAL_WIRES):
        idx = layer_idx * TOTAL_WIRES * 3 + wire * 3
        qc.rz(params[idx],     wire)
        qc.ry(params[idx + 1], wire)
        qc.rz(params[idx + 2], wire)

    # Part B: Entangling CNOT gates (ring pattern)
    # Qubit 0 controls qubit 1, qubit 1 controls qubit 2, ..., last controls first
    # This creates quantum correlations between all qubits
    for wire in range(TOTAL_WIRES):
        qc.cx(wire, (wire + 1) % TOTAL_WIRES)

    # Part C: Cross-register entanglement (KEY for kinship!)
    # Directly entangle person1's qubits with person2's qubits
    # This lets the circuit compare the two faces at a quantum level
    for i in range(N_QUBITS):
        qc.cx(i, i + N_QUBITS)   # qubit i (P1) ↔ qubit i+4 (P2)


# ─────────────────────────────────────────────────────────────────────────────
# FULL QUANTUM KINSHIP CIRCUIT (parameterized)
# ─────────────────────────────────────────────────────────────────────────────

def build_kinship_circuit(n_layers=N_LAYERS):
    """
    Build the full parameterized kinship circuit.

    Returns:
        qc            : QuantumCircuit with input + weight parameters
        input_params  : ParameterVector for feature encoding (2 * N_QUBITS)
        weight_params : ParameterVector for trainable VQC weights
    """
    n_weight_params = n_layers * TOTAL_WIRES * 3
    input_params  = ParameterVector('x', 2 * N_QUBITS)   # 4 features per person × 2
    weight_params = ParameterVector('θ', n_weight_params)

    qc = QuantumCircuit(TOTAL_WIRES)

    # ── Step 1: Encode both faces into their qubits ────────────────────────
    for i in range(N_QUBITS):
        qc.ry(input_params[i], i)                      # Person 1: qubits 0-3
    for i in range(N_QUBITS):
        qc.ry(input_params[N_QUBITS + i], N_QUBITS + i)  # Person 2: qubits 4-7

    qc.barrier()

    # ── Step 2: Run N_LAYERS of the VQC ───────────────────────────────────
    for layer in range(n_layers):
        vqc_layer(qc, weight_params, layer)
        if layer < n_layers - 1:
            qc.barrier()

    return qc, input_params, weight_params


def run_kinship_circuit(features1, features2, params, n_layers=N_LAYERS):
    """
    Run the kinship circuit with concrete parameter values.

    Args:
        features1 : encoded face features of Person 1, shape (N_QUBITS,)
        features2 : encoded face features of Person 2, shape (N_QUBITS,)
        params    : trainable VQC parameters, flat array

    Returns:
        Expectation value of PauliZ on qubit 0.
        Range: [-1, +1].   +1 ≈ kin,  -1 ≈ not kin  (after training)
    """
    qc, input_p, weight_p = build_kinship_circuit(n_layers)

    # Bind all parameters
    input_vals = np.concatenate([features1, features2])
    param_dict = {}
    for i, p in enumerate(input_p):
        param_dict[p] = float(input_vals[i])
    for i, p in enumerate(weight_p):
        param_dict[p] = float(params[i])

    bound_qc = qc.assign_parameters(param_dict)

    # Get statevector and compute ⟨Z⟩ on qubit 0
    sv = Statevector.from_instruction(bound_qc)
    # PauliZ on qubit 0: construct the operator
    # In Qiskit, SparsePauliOp labels are right-to-left, so qubit 0 is rightmost
    z_label = 'I' * (TOTAL_WIRES - 1) + 'Z'
    op = SparsePauliOp(z_label)
    expval = sv.expectation_value(op).real

    return expval


# ─────────────────────────────────────────────────────────────────────────────
# ALTERNATIVE CIRCUIT: Swap Test (measures fidelity directly)
# ─────────────────────────────────────────────────────────────────────────────
# The SWAP test is a classic quantum algorithm to measure how similar
# two quantum states are WITHOUT looking at them directly.
#
# It uses an ancilla (helper) qubit:
#   P(ancilla = |0⟩) = (1 + |⟨ψ1|ψ2⟩|²) / 2
#   → Fidelity F = 2·P(|0⟩) - 1

def build_swap_test_circuit(features1, features2):
    """
    SWAP test to measure fidelity between two face states.
    No trainable parameters — this is a fixed quantum algorithm.

    Wire layout:
        Qubit 0           → ancilla qubit
        Qubits 1..N       → Person 1's qubits
        Qubits N+1..2N    → Person 2's qubits

    Returns:
        Probability of measuring ancilla as |0⟩
    """
    total = 1 + 2 * N_QUBITS   # 1 ancilla + 2 × N_QUBITS data qubits
    qc = QuantumCircuit(total)

    ancilla = 0
    wires1 = range(1, N_QUBITS + 1)
    wires2 = range(N_QUBITS + 1, 2 * N_QUBITS + 1)

    # Encode both face states
    for i, w in enumerate(wires1):
        qc.ry(features1[i], w)
    for i, w in enumerate(wires2):
        qc.ry(features2[i], w)

    # SWAP test algorithm:
    qc.h(ancilla)                                    # put ancilla in superposition
    for w1, w2 in zip(wires1, wires2):
        qc.cswap(ancilla, w1, w2)                    # controlled-SWAP
    qc.h(ancilla)                                    # interfere

    return qc


def compute_fidelity(features1, features2):
    """
    Computes quantum fidelity F = |⟨ψ1|ψ2⟩|² using the SWAP test.
    F=1 means identical states, F=0 means orthogonal (opposite) states.
    """
    qc = build_swap_test_circuit(features1, features2)
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities([0])   # probabilities of ancilla qubit (qubit 0)
    p0 = probs[0]                   # probability of ancilla = |0⟩
    fidelity = 2 * p0 - 1           # extract fidelity from measurement
    return float(fidelity)


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def init_params(n_layers=N_LAYERS, seed=42):
    """
    Initialize VQC parameters randomly.
    Total params: n_layers × TOTAL_WIRES × 3  (3 angles per qubit per layer).
    """
    rng = np.random.default_rng(seed)
    n_params = n_layers * TOTAL_WIRES * 3
    # Small random initialization — avoids barren plateaus (flat gradient regions)
    return rng.uniform(-np.pi/4, np.pi/4, size=n_params)


# ── Main demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("STEP 3: Variational Quantum Circuit (Qiskit)")
    print("="*55)

    # Use the same fake features from step 2
    def compress(raw, n):
        chunk = len(raw) // n
        out = np.array([np.mean(raw[i*chunk:(i+1)*chunk]) for i in range(n)])
        out = np.pi * (out - out.min()) / (out.max() - out.min() + 1e-8)
        return out

    rng = np.random.default_rng(0)
    emb1 = rng.standard_normal(128); emb1 /= np.linalg.norm(emb1)
    emb2 = rng.standard_normal(128); emb2 /= np.linalg.norm(emb2)
    emb3 = emb1 + rng.standard_normal(128)*0.05   # emb3 ≈ emb1 (simulate sibling)
    emb3 /= np.linalg.norm(emb3)

    f1 = compress(emb1, N_QUBITS)
    f2 = compress(emb2, N_QUBITS)
    f3 = compress(emb3, N_QUBITS)

    # ── Test 1: VQC with random params ────────────────────────────────────────
    params = init_params(N_LAYERS)
    n_total = N_LAYERS * TOTAL_WIRES * 3
    print(f"\nVQC parameter shape: ({N_LAYERS} layers × {TOTAL_WIRES} qubits × 3 angles)")
    print(f"  = {n_total} trainable parameters total")

    score_1v2 = run_kinship_circuit(f1, f2, params)
    score_1v3 = run_kinship_circuit(f1, f3, params)
    print(f"\nVQC score (Person1 vs Person2, strangers) : {score_1v2:.4f}")
    print(f"VQC score (Person1 vs Person3, siblings)  : {score_1v3:.4f}")
    print("(These are random before training — training is in step 4!)")

    # ── Test 2: SWAP test fidelity ─────────────────────────────────────────────
    fid_1v2 = compute_fidelity(f1, f2)
    fid_1v3 = compute_fidelity(f1, f3)
    print(f"\nSWAP test fidelity (Person1 vs Person2) : {fid_1v2:.4f}")
    print(f"SWAP test fidelity (Person1 vs Person3) : {fid_1v3:.4f}")
    print("(Higher fidelity = more similar quantum states = more likely kin)")

    # ── Draw the circuit ───────────────────────────────────────────────────────
    qc, _, _ = build_kinship_circuit(n_layers=1)
    print("\nKinship VQC circuit (1 layer shown):")
    print(qc.draw('text', fold=100))

    print("\n✅ VQC ready! Move on to step4_training.py")
