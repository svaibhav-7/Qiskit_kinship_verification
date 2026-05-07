import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from .quantum_core import build_kinship_vqc, run_kinship_circuit, compress_embedding
from qiskit.quantum_info import SparsePauliOp

class MockCNN:
    """Placeholder for a real face feature extractor."""
    def __init__(self, embedding_dim=128, seed=0):
        self.embedding_dim = embedding_dim
        self._rng = np.random.default_rng(seed)

    def extract(self, image_or_id):
        if isinstance(image_or_id, int):
            rng = np.random.default_rng(image_or_id)
            raw = rng.standard_normal(self.embedding_dim)
        else:
            raw = self._rng.standard_normal(self.embedding_dim)
        return raw / (np.linalg.norm(raw) + 1e-8)

class FaceFeatureExtractor:
    """Real feature extractor using pretrained ResNet-18."""
    def __init__(self):
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Identity()
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, img_path):
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0)
        emb = self.model(tensor).squeeze().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)

class QuantumKinshipModel:
    def __init__(self, n_qubits=4, n_layers=3, feature_extractor=None, seed=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.extractor = feature_extractor or MockCNN()
        
        self.qc, self.inp_p, self.wgt_p = build_kinship_vqc(n_qubits, n_layers)
        self.z_op = SparsePauliOp('I' * (2 * n_qubits - 1) + 'Z')
        
        rng = np.random.default_rng(seed)
        n_params = n_layers * (2 * n_qubits) * 3
        self.params = rng.uniform(-np.pi/4, np.pi/4, size=n_params)

    def predict(self, image1, image2):
        e1 = self.extractor.extract(image1)
        e2 = self.extractor.extract(image2)
        f1 = compress_embedding(e1, self.n_qubits)
        f2 = compress_embedding(e2, self.n_qubits)
        
        score = run_kinship_circuit(self.qc, self.inp_p, self.wgt_p, f1, f2, self.params, self.z_op)
        return score

    def save_weights(self, path):
        np.save(path, self.params)

    def load_weights(self, path):
        self.params = np.load(path)
