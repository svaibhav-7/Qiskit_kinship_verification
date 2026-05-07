from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
import numpy as np

def build_kinship_vqc(n_qubits, n_layers):
    """Build the parameterized kinship VQC."""
    total_wires = 2 * n_qubits
    n_params = n_layers * total_wires * 3
    
    inp = ParameterVector('x', total_wires)
    wgt = ParameterVector('w', n_params)
    qc  = QuantumCircuit(total_wires)

    # Encode both persons
    for i in range(n_qubits):
        qc.ry(inp[i], i)
    for i in range(n_qubits):
        qc.ry(inp[n_qubits + i], n_qubits + i)

    # VQC layers
    for layer in range(n_layers):
        base = layer * total_wires * 3
        for wire in range(total_wires):
            idx = base + wire * 3
            qc.rz(wgt[idx], wire)
            qc.ry(wgt[idx+1], wire)
            qc.rz(wgt[idx+2], wire)
        for wire in range(total_wires):
            qc.cx(wire, (wire + 1) % total_wires)
        for i in range(n_qubits):
            qc.cx(i, i + n_qubits)

    return qc, inp, wgt

def run_kinship_circuit(qc, inp_params, wgt_params, f1, f2, weights, z_op):
    """Evaluate ⟨Z⟩ on qubit 0 for a pair of feature vectors."""
    vals = np.concatenate([f1, f2])
    param_dict = {}
    for i, p in enumerate(inp_params):
        param_dict[p] = float(vals[i])
    for i, p in enumerate(wgt_params):
        param_dict[p] = float(weights[i])
    
    bound_qc = qc.assign_parameters(param_dict)
    sv = Statevector.from_instruction(bound_qc)
    return float(sv.expectation_value(z_op).real)

def compress_embedding(emb, n_qubits):
    """Compress high-dim embedding into n_qubits rotation angles."""
    chunk = len(emb) // n_qubits
    out = np.array([np.mean(emb[i*chunk:(i+1)*chunk]) for i in range(n_qubits)])
    lo, hi = out.min(), out.max()
    return np.pi * (out - lo) / (hi - lo + 1e-8)
