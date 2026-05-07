"""
=============================================================
QUANTUM KINSHIP PIPELINE — STEP 2: QUANTUM ENCODING
=============================================================
This file covers HOW we convert a face embedding vector
(output of a classical CNN) into a quantum state |ψ_z⟩.

Pipeline position:
    Image → CNN → Embedding z → [QUANTUM ENCODING] → |ψ_z⟩ → VQC → Measure
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: What is a face embedding?
# ─────────────────────────────────────────────────────────────────────────────
# A classical face recognition CNN (like FaceNet or ArcFace) takes an image
# and outputs a vector of floats — e.g., 128 numbers.
# Each number captures something about the face (jawline, eye spacing, etc.)
#
# Example: z = [0.12, -0.45, 0.78, 0.03, ...]   ← 128 numbers
#
# We need to encode these 128 numbers INTO qubits.
# We'll use ANGLE ENCODING: each number becomes a rotation angle on a qubit.
# ─────────────────────────────────────────────────────────────────────────────

# ── Configuration ─────────────────────────────────────────────────────────────
EMBEDDING_DIM = 128      # size of the face embedding vector from the CNN
N_QUBITS      = 4        # number of qubits (we compress 128 → 4 for simulation)
                         # In a real quantum computer you'd use more qubits


# ─────────────────────────────────────────────────────────────────────────────
# ENCODING METHOD 1: Angle Encoding (simplest, most intuitive)
# ─────────────────────────────────────────────────────────────────────────────
# Each feature value x_i becomes the rotation angle of one qubit.
# RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
#
# Think of a qubit as a globe (Bloch sphere).
# RY rotates the qubit up/down. The angle decides how much 0 vs 1 it is.
# ─────────────────────────────────────────────────────────────────────────────

def angle_encoding(qc, features, n_qubits, offset=0):
    """
    Encode a feature vector into qubits using rotation angles.

    Args:
        qc       : QuantumCircuit to add gates to
        features : 1D array of floats, length = n_qubits
                   (each value should be in range [0, π] for best coverage)
        n_qubits : number of qubits
        offset   : starting qubit index (for multi-register encoding)
    """
    for i in range(n_qubits):
        qc.ry(features[i], offset + i)   # rotate qubit i by features[i] radians


# ─────────────────────────────────────────────────────────────────────────────
# ENCODING METHOD 2: Amplitude Encoding (more powerful, fits more data)
# ─────────────────────────────────────────────────────────────────────────────
# n qubits can store 2^n amplitudes.
# So 4 qubits can encode a 2^4 = 16-dimensional vector.
# 7 qubits can encode 2^7 = 128-dimensional vector (your full embedding!)
#
# The state becomes: |ψ⟩ = Σ x_i |i⟩  (normalized)
# ─────────────────────────────────────────────────────────────────────────────

def amplitude_encoding(qc, features, n_qubits):
    """
    Encode a normalized feature vector as quantum state amplitudes.
    Uses Qiskit's initialize method.

    Args:
        qc       : QuantumCircuit to add gates to
        features : 1D array of floats, length must equal 2^n_qubits
                   Will be L2-normalized automatically
        n_qubits : number of qubits
    """
    # Normalize so |α|² + |β|² + ... = 1
    norm = np.linalg.norm(features)
    if norm > 1e-10:
        features = features / norm
    qc.initialize(features, range(n_qubits))


# ── Build a test circuit using angle encoding ─────────────────────────────────

def encoding_circuit(embedding, n_qubits=N_QUBITS):
    """
    Takes a face embedding, encodes it into qubits, returns state vector.

    Args:
        embedding : array of shape (N_QUBITS,) — the compressed face features
    Returns:
        state vector: complex array of length 2^N_QUBITS
    """
    qc = QuantumCircuit(n_qubits)

    # Step 1: encode the embedding into qubits
    angle_encoding(qc, embedding, n_qubits)

    # Step 2: return the full quantum state vector (for inspection)
    sv = Statevector.from_instruction(qc)
    return sv


# ── Simulate a face embedding (placeholder for real CNN output) ───────────────

def make_fake_embedding(seed=42):
    """
    Creates a fake face embedding vector.
    In your real pipeline this comes from a CNN like FaceNet or ArcFace.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(EMBEDDING_DIM).astype(float)
    # L2 normalize — standard practice for face embeddings
    normalized = raw / np.linalg.norm(raw)
    return normalized


def compress_embedding(embedding, n_qubits):
    """
    Compress a 128-dim embedding down to n_qubits features.
    Simple approach: split into chunks, take mean of each chunk.

    In a real system you'd use a trainable classical layer here.
    """
    chunk_size = len(embedding) // n_qubits
    compressed = np.array([
        np.mean(embedding[i*chunk_size:(i+1)*chunk_size])
        for i in range(n_qubits)
    ])
    # Scale to [0, π] so RY gates cover the full Bloch sphere
    compressed = np.pi * (compressed - compressed.min()) / (compressed.max() - compressed.min() + 1e-8)
    return compressed


# ── Main demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("STEP 2: Quantum Encoding of Face Embeddings")
    print("="*55)

    # 1. Create two fake face embeddings (Person A and Person B)
    embedding_A = make_fake_embedding(seed=1)   # person A's face
    embedding_B = make_fake_embedding(seed=2)   # person B's face

    print(f"\nFace embedding shape  : {embedding_A.shape}")
    print(f"First 5 values (A)    : {embedding_A[:5].round(3)}")
    print(f"L2 norm (should be 1) : {np.linalg.norm(embedding_A):.4f}")

    # 2. Compress from 128-dim → N_QUBITS features
    features_A = compress_embedding(embedding_A, N_QUBITS)
    features_B = compress_embedding(embedding_B, N_QUBITS)

    print(f"\nCompressed to {N_QUBITS} features:")
    print(f"  Person A: {features_A.round(3)}")
    print(f"  Person B: {features_B.round(3)}")

    # 3. Encode into quantum state
    state_A = encoding_circuit(features_A)
    state_B = encoding_circuit(features_B)

    print(f"\nQuantum state |ψ_A⟩ (complex amplitudes):")
    amplitudes = state_A.data
    for i, amp in enumerate(amplitudes):
        prob = abs(amp)**2
        basis = format(i, f'0{N_QUBITS}b')   # e.g. '0011'
        print(f"  |{basis}⟩ : amplitude={amp.real:.4f}, probability={prob:.4f}")

    print(f"\nSum of probabilities = {sum(abs(a)**2 for a in amplitudes):.6f}  (should be 1.0)")

    # 4. Draw the encoding circuit
    qc_demo = QuantumCircuit(N_QUBITS)
    angle_encoding(qc_demo, features_A, N_QUBITS)
    print("\nEncoding circuit diagram:")
    print(qc_demo.draw('text'))

    # 5. Classical similarity (cosine) for reference
    cos_sim = np.dot(embedding_A, embedding_B)
    print(f"\nClassical cosine similarity (A vs B): {cos_sim:.4f}")
    print("(This is what a normal face recognition system would use)")
    print("\n✅ Encoding works! Move on to step3_vqc.py")
