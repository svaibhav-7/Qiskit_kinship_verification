import torch
import torch.nn as nn
from torchvision import models
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_QUBITS_PER_PERSON = 8   # 8 qubits per person = 16 total
N_QUBITS_TOTAL      = 2 * N_QUBITS_PER_PERSON
N_FEATURES          = N_QUBITS_PER_PERSON  # 1 feature per qubit (Angle Encoding)
N_LAYERS            = 3
N_PARAMS            = N_LAYERS * N_QUBITS_TOTAL * 3

# ═══════════════════════════════════════════════════════════════════════════════
# 1. QUANTUM CIRCUIT DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

def create_kinship_qnn():
    """Creates an EstimatorQNN for hybrid training."""
    # Parameters
    input_params = ParameterVector('x', N_QUBITS_TOTAL)
    weight_params = ParameterVector('θ', N_PARAMS)
    
    qc = QuantumCircuit(N_QUBITS_TOTAL)
    
    # --- Step 1: Feature Encoding (Angle Encoding) ---
    # Person 1 (0-7), Person 2 (8-15)
    for i in range(N_QUBITS_TOTAL):
        qc.ry(input_params[i], i)
    
    qc.barrier()
    
    # --- Step 2: Variational Layers ---
    for layer in range(N_LAYERS):
        base = layer * N_QUBITS_TOTAL * 3
        # Trainable rotations
        for wire in range(N_QUBITS_TOTAL):
            idx = base + wire * 3
            qc.rz(weight_params[idx], wire)
            qc.ry(weight_params[idx+1], wire)
            qc.rz(weight_params[idx+2], wire)
        
        # Entanglement (Circular CNOT)
        for wire in range(N_QUBITS_TOTAL):
            qc.cx(wire, (wire + 1) % N_QUBITS_TOTAL)
            
        # Cross-register entanglement (Direct Comparison)
        for i in range(N_QUBITS_PER_PERSON):
            qc.cx(i, i + N_QUBITS_PER_PERSON)
        
        qc.barrier()
    
    # Observable: PauliZ on qubit 0 (Kinship Score)
    observable = SparsePauliOp('I' * (N_QUBITS_TOTAL - 1) + 'Z')
    
    # Create QNN
    qnn = EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params
    )
    
    return qnn

# ═══════════════════════════════════════════════════════════════════════════════
# 2. HYBRID MODEL CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class HybridQuantumKinship(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Classical: ResNet-18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1]) # Remove FC layer
        
        # Bottleneck/Compression: 512 -> N_QUBITS_PER_PERSON
        # This layer learns how to compress face data for the Quantum Circuit
        self.compression = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, N_QUBITS_PER_PERSON),
            nn.Sigmoid() # Scale to [0, 1] then we multiply by pi
        )
        
        # Quantum Layer
        qnn = create_kinship_qnn()
        self.quantum_layer = TorchConnector(qnn)
        
    def forward(self, img1, img2):
        # 1. Classical Feature Extraction
        feat1 = self.cnn(img1)
        feat2 = self.cnn(img2)
        
        # 2. Compression to Quantum Input
        # We want features in range [0, pi] for RY gates
        q_input1 = self.compression(feat1) * np.pi
        q_input2 = self.compression(feat2) * np.pi
        
        # 3. Concatenate for Quantum Circuit (16-dim input)
        q_input = torch.cat([q_input1, q_input2], dim=1)
        
        # 4. Quantum Forward Pass
        # Returns score in [-1, 1]
        score = self.quantum_layer(q_input)
        
        return score

# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class ContrastiveKinshipLoss(nn.Module):
    """
    Forces Kinship scores towards 1.0 and Non-Kinship towards -1.0.
    Similar to Contrastive/Margin loss.
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, score, target):
        # target: 1 for kin, -1 for non-kin
        # loss = mean( (score - target)^2 )
        loss = torch.mean((score - target)**2)
        return loss
