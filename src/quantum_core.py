from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np

def build_swap_test_circuit(n_qubits, z1, z2):
    """
    Builds a Quantum SWAP Test circuit to measure the overlap (fidelity) 
    between two angle-encoded states |psi_z1> and |psi_z2>.
    
    Args:
        n_qubits (int): Number of qubits representing each face embedding.
        z1 (array-like): Angle encoding parameters for Person 1 (length n_qubits).
        z2 (array-like): Angle encoding parameters for Person 2 (length n_qubits).
        
    Returns:
        qc (QuantumCircuit): The complete Qiskit QuantumCircuit.
    """
    # 2N + 1 qubits:
    # - Qubit 0: Ancilla qubit
    # - Qubits 1 ... N: Register 1 (Person 1 features)
    # - Qubits N+1 ... 2N: Register 2 (Person 2 features)
    total_qubits = 2 * n_qubits + 1
    qc = QuantumCircuit(total_qubits, 1)
    
    # 1. State preparation (Ry angle encoding)
    for i in range(n_qubits):
        qc.ry(float(z1[i]), 1 + i)
        qc.ry(float(z2[i]), 1 + n_qubits + i)
        
    # 2. SWAP Test
    qc.h(0) # Put ancilla in superposition
    for i in range(n_qubits):
        # Controlled-SWAP: swap corresponding qubits in reg 1 and reg 2 if ancilla is 1
        qc.cswap(0, 1 + i, 1 + n_qubits + i)
    qc.h(0) # Bring ancilla out of superposition
    
    # 3. Measurement of the ancilla qubit
    qc.measure(0, 0)
    
    return qc

def simulate_swap_test(z1, z2, shots=1024):
    """
    Simulates the SWAP test circuit on AerSimulator to obtain the state fidelity.
    
    Args:
        z1 (array-like): Angle parameters for Person 1.
        z2 (array-like): Angle parameters for Person 2.
        shots (int): Number of simulation shots.
        
    Returns:
        fidelity (float): Overlap value in range [0, 1].
    """
    n_qubits = len(z1)
    qc = build_swap_test_circuit(n_qubits, z1, z2)
    
    simulator = AerSimulator()
    job = simulator.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # Get count of the state '0'
    count_0 = counts.get('0', 0)
    p_0 = count_0 / shots
    
    # Overlap fidelity = 2 * P(0) - 1
    fidelity = 2 * p_0 - 1.0
    return max(0.0, min(1.0, fidelity))

def simulate_swap_test_batch(z1_batch, z2_batch, shots=1024):
    """
    Runs SWAP test simulation for a batch of feature pairs in parallel.
    
    Args:
        z1_batch (np.ndarray): (Batch, n_qubits) array of angles.
        z2_batch (np.ndarray): (Batch, n_qubits) array of angles.
        shots (int): Number of shots.
        
    Returns:
        fidelities (np.ndarray): (Batch, 1) array of overlap values.
    """
    batch_size = len(z1_batch)
    n_qubits = z1_batch.shape[1]
    
    circuits = []
    for i in range(batch_size):
        qc = build_swap_test_circuit(n_qubits, z1_batch[i], z2_batch[i])
        circuits.append(qc)
        
    simulator = AerSimulator()
    job = simulator.run(circuits, shots=shots)
    results = job.result()
    
    fidelities = []
    for i in range(batch_size):
        counts = results.get_counts(i)
        count_0 = counts.get('0', 0)
        p_0 = count_0 / shots
        fidelity = 2 * p_0 - 1.0
        fidelities.append([max(0.0, min(1.0, fidelity))])
        
    return np.array(fidelities)
