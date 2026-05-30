import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import warnings

from .quantum_core import simulate_swap_test_batch

class FaceFeatureExtractor:
    """
    Classical CNN Face Feature Extractor.
    Extracts high-quality facial embeddings using FaceNet (InceptionResnetV1) pretrained on VGGFace2.
    Gracefully falls back to torchvision ResNet-18 if FaceNet fails or is unavailable.
    """
    def __init__(self, use_resnet_fallback=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fallback = use_resnet_fallback
        
        # Define standard image transform (resize, convert to tensor, normalize)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        
        if not self.fallback:
            try:
                from facenet_pytorch import InceptionResnetV1
                print("Initializing FaceNet (InceptionResnetV1) pretrained on VGGFace2...")
                self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                self.embedding_dim = 512
                print("FaceNet feature extractor successfully initialized.")
            except Exception as e:
                warnings.warn(f"Failed to initialize FaceNet ({e}). Falling back to torchvision ResNet-18.")
                self.fallback = True
                
        if self.fallback:
            print("Initializing torchvision ResNet-18 pretrained on ImageNet...")
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device)
            self.model.fc = nn.Identity()  # Remove classifier to output 512-dim features
            self.model.eval()
            self.embedding_dim = 512
            print("ResNet-18 feature extractor successfully initialized.")

    @torch.no_grad()
    def extract(self, img_path):
        """
        Extracts L2-normalized 512-dimensional embedding for a single face image.
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        emb = self.model(tensor).squeeze()
        emb_np = emb.cpu().numpy()
        norm = np.linalg.norm(emb_np) + 1e-8
        return emb_np / norm

    @torch.no_grad()
    def extract_batch(self, img_paths):
        """
        Extracts L2-normalized embeddings for a batch of face images.
        """
        tensors = []
        for path in img_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")
            img = Image.open(path).convert('RGB')
            tensors.append(self.transform(img))
            
        batch = torch.stack(tensors).to(self.device)
        embs = self.model(batch)
        embs_np = embs.cpu().numpy()
        norms = np.linalg.norm(embs_np, axis=1, keepdims=True) + 1e-8
        return embs_np / norms


class HybridKinshipClassifier(nn.Module):
    """
    Hybrid Classical-Quantum model for kinship verification using SWAP Test.
    Projects classical face embeddings into lower-dimensional qubit angle parameters,
    and compares them using quantum statevector overlap fidelity.
    """
    def __init__(self, n_qubits=8):
        super().__init__()
        self.n_qubits = n_qubits
        
        # Projection MLP with relation conditioning: (512 + 4) -> 128 -> n_qubits
        self.projection = nn.Sequential(
            nn.Linear(512 + 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_qubits)
        )

    def forward(self, emb1, emb2, rels):
        """
        Differentiable analytical forward pass of the quantum overlap (fidelity).
        Runs instantly in PyTorch and allows for fast gradient descent.
        """
        # Concatenate 512-dim facial features with 4-dim one-hot relation vector
        x1 = torch.cat([emb1, rels], dim=1)
        x2 = torch.cat([emb2, rels], dim=1)
        
        # Map features into quantum parameters (angles) in [-pi, pi]
        z1 = torch.tanh(self.projection(x1)) * np.pi
        z2 = torch.tanh(self.projection(x2)) * np.pi
        
        # Exact statevector fidelity of Ry angle-encoded qubits:
        # F = prod(cos((z1 - z2)/2) ** 2)
        cos_diff = torch.cos((z1 - z2) / 2.0)
        fidelity = torch.prod(cos_diff ** 2, dim=1, keepdim=True)
        return fidelity

    def forward_qiskit(self, emb1, emb2, rels, shots=1024):
        """
        Executes actual Qiskit SWAP test quantum circuits on AerSimulator
        using the trained classical projection weights and relation context.
        """
        self.eval()
        with torch.no_grad():
            x1 = torch.cat([emb1, rels], dim=1)
            x2 = torch.cat([emb2, rels], dim=1)
            z1 = torch.tanh(self.projection(x1)) * np.pi
            z2 = torch.tanh(self.projection(x2)) * np.pi
            
        z1_np = z1.cpu().numpy()
        z2_np = z2.cpu().numpy()
        
        # Run Qiskit batch simulation
        fidelities = simulate_swap_test_batch(z1_np, z2_np, shots=shots)
        return torch.tensor(fidelities, dtype=torch.float32)
