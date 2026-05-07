"""
=============================================================
QUANTUM KINSHIP PIPELINE — STEP 1: SETUP & ENVIRONMENT CHECK
=============================================================
Run this file FIRST to make sure everything is installed correctly.

Install dependencies with:
    pip install qiskit qiskit-aer qiskit-machine-learning torch numpy matplotlib scipy Pillow
"""

# ── Standard imports ──────────────────────────────────────────────────────────
import sys
import numpy as np

# ── Check Qiskit ──────────────────────────────────────────────────────────────
try:
    import qiskit
    print(f"✅ Qiskit version           : {qiskit.__version__}")
except ImportError:
    print("❌ Qiskit not found. Run: pip install qiskit")
    sys.exit(1)

# ── Check Qiskit Aer ──────────────────────────────────────────────────────────
try:
    import qiskit_aer
    print(f"✅ Qiskit Aer version       : {qiskit_aer.__version__}")
except ImportError:
    print("❌ Qiskit Aer not found. Run: pip install qiskit-aer")
    sys.exit(1)

# ── Check Qiskit Machine Learning ────────────────────────────────────────────
try:
    import qiskit_machine_learning
    print(f"✅ Qiskit ML version        : {qiskit_machine_learning.__version__}")
except ImportError:
    print("❌ Qiskit ML not found. Run: pip install qiskit-machine-learning")
    sys.exit(1)

# ── Check PyTorch ─────────────────────────────────────────────────────────────
try:
    import torch
    print(f"✅ PyTorch version          : {torch.__version__}")
except ImportError:
    print("❌ PyTorch not found. Run: pip install torch")
    sys.exit(1)

# ── Check Matplotlib ──────────────────────────────────────────────────────────
try:
    import matplotlib
    print(f"✅ Matplotlib version       : {matplotlib.__version__}")
except ImportError:
    print("⚠️  Matplotlib not found (optional). Run: pip install matplotlib")

print("\n" + "="*55)
print("QUICK SANITY TEST — Hello Quantum World")
print("="*55)

# ── Build a tiny 2-qubit circuit to confirm Qiskit works ─────────────────────
#    This is the simplest possible quantum circuit:
#    1. Put qubit 0 into superposition with a Hadamard gate
#    2. Entangle qubit 0 and qubit 1 with a CNOT gate
#    3. Measure probabilities of all 4 states: |00⟩, |01⟩, |10⟩, |11⟩

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Create the Bell state circuit
qc = QuantumCircuit(2)
qc.h(0)          # qubit 0: |0⟩ → (|0⟩ + |1⟩)/√2
qc.cx(0, 1)      # entangle: if qubit0=1 then flip qubit1

# Get the statevector and compute probabilities
sv = Statevector.from_instruction(qc)
probs = sv.probabilities()

print(f"\nBell state outcome probabilities:")
print(f"  |00⟩ : {probs[0]:.3f}   (expect ~0.5)")
print(f"  |01⟩ : {probs[1]:.3f}   (expect ~0.0)")
print(f"  |10⟩ : {probs[2]:.3f}   (expect ~0.0)")
print(f"  |11⟩ : {probs[3]:.3f}   (expect ~0.5)")

# ── Draw the circuit ──────────────────────────────────────────────────────────
print("\nCircuit diagram:")
print(qc.draw('text'))

# ── Also run on AerSimulator with shots to show sampling ─────────────────────
from qiskit_aer import AerSimulator

qc_meas = qc.copy()
qc_meas.measure_all()

simulator = AerSimulator()
result = simulator.run(qc_meas, shots=1024).result()
counts = result.get_counts()

print(f"\nShot-based simulation (1024 shots):")
for state, count in sorted(counts.items()):
    print(f"  |{state}⟩ : {count} counts ({count/1024*100:.1f}%)")

print("\n✅ All good! Move on to step2_quantum_encoding.py")
