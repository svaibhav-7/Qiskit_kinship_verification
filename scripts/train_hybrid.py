"""
=============================================================
HYBRID QUANTUM KINSHIP — END-TO-END TRAINING
=============================================================
Trains the CNN and Quantum Circuit together.
Target: 85-90%+ accuracy on KinFaceW.
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import scipy.io

from src.hybrid_model import HybridQuantumKinship, ContrastiveKinshipLoss

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KFW1 = os.path.join(BASE, "KinFaceW-I", "KinFaceW-I")
KFW2 = os.path.join(BASE, "KinFaceW-II")

# ── DATASET CLASS ─────────────────────────────────────────────────────────────

class KinshipDataset(Dataset):
    def __init__(self, root, transform=None, max_pairs=None):
        self.root = root
        self.transform = transform
        self.pairs = []
        
        relations = ['fd', 'fs', 'md', 'ms']
        rel_dirs = {'fd': 'father-dau', 'fs': 'father-son',
                    'md': 'mother-dau', 'ms': 'mother-son'}
        
        for rel in relations:
            mat_path = os.path.join(root, "meta_data", f"{rel}_pairs.mat")
            if not os.path.exists(mat_path): continue
            
            data = scipy.io.loadmat(mat_path)
            mat_pairs = data['pairs']
            img_dir = os.path.join(root, "images", rel_dirs[rel])
            
            count = 0
            for row in mat_pairs:
                label = int(row[1].flat[0]) # 1=kin, 0=non-kin
                p1 = os.path.join(img_dir, str(row[2].flat[0]))
                p2 = os.path.join(img_dir, str(row[3].flat[0]))
                
                if os.path.exists(p1) and os.path.exists(p2):
                    self.pairs.append((p1, p2, 1.0 if label == 1 else -1.0))
                    count += 1
                if max_pairs and count >= max_pairs: break
        
        print(f"Loaded {len(self.pairs)} pairs from {root}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1 = Image.open(p1).convert('RGB')
        img2 = Image.open(p2).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, torch.tensor([label], dtype=torch.float32)

# ── TRAINING LOOP ─────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Setup Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = KinshipDataset(KFW2, transform=transform)
    test_ds = KinshipDataset(KFW1, transform=transform, max_pairs=100) # subset for faster eval
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    # 2. Setup Model
    model = HybridQuantumKinship().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = ContrastiveKinshipLoss()
    
    print("\nStarting Hybrid Training (Classical + Quantum)...")
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for i, (img1, img2, labels) in enumerate(train_loader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            scores = model(img1, img2)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            preds = torch.where(scores >= 0, 1.0, -1.0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {100*correct/total:.1f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for img1, img2, labels in test_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                scores = model(img1, img2)
                preds = torch.where(scores >= 0, 1.0, -1.0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        print(f"--- End of Epoch {epoch+1} | Train Acc: {100*correct/total:.1f}% | Val Acc: {100*val_correct/val_total:.1f}% ---")
        
        # Save checkpoint
        torch.save(model.state_state_dict(), f"weights/hybrid_kinship_e{epoch+1}.pth")

if __name__ == "__main__":
    train()
