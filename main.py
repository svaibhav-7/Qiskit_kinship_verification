import os
import torch
import numpy as np

from src.models import FaceFeatureExtractor, HybridKinshipClassifier
from src.quantum_core import simulate_swap_test
from src.data_loaders import get_relation_category

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(project_root, "weights", "hybrid_kinship.pt")
    KFW1 = os.path.join(project_root, "KinFaceW-I", "KinFaceW-I")
    
    print("=" * 60)
    print("      QUANTUM SWAP TEST KINSHIP VERIFICATION DEMO")
    print("=" * 60)
    
    # 1. Initialize models
    print("[1/3] Initializing models...")
    extractor = FaceFeatureExtractor()
    
    n_qubits = 8
    
    # Load trained weights if available
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path)
            # Find qubits count dynamically from the last linear layer in the sequential projection (handling both 3-module and 4-module sequential projection)
            if 'projection.3.bias' in state_dict:
                n_qubits = state_dict['projection.3.bias'].shape[0]
            elif 'projection.2.bias' in state_dict:
                n_qubits = state_dict['projection.2.bias'].shape[0]
            print(f"  [INFO] Detected checkpoint configuration: n_qubits={n_qubits}")
            model = HybridKinshipClassifier(n_qubits=n_qubits)
            model.load_state_dict(state_dict)
            print(f"  [OK] Successfully loaded trained weights from: {weights_path}")
        except Exception as e:
            print(f"  [WARN] Failed to load trained weights ({e}). Using default/random initialization.")
            model = HybridKinshipClassifier(n_qubits=n_qubits)
    else:
        print("  [INFO] No trained model weights found. Using default/random initialization.")
        model = HybridKinshipClassifier(n_qubits=n_qubits)
        
    model.eval()
    print("-" * 60)
    
    # 2. Load a sample pair from KinFaceW-I test set
    print("[2/3] Loading sample pair from KinFaceW-I...")
    
    sample_dir = os.path.join(KFW1, "images", "father-dau")
    if not os.path.exists(sample_dir):
        print(f"  [ERROR] Dataset directory not found: {sample_dir}")
        print("  Please make sure datasets are downloaded and placed in the project root.")
        return
        
    images = sorted([img for img in os.listdir(sample_dir) if img.endswith('.jpg')])
    if len(images) < 2:
        print("  [ERROR] Not enough images in father-dau directory.")
        return
        
    p1 = os.path.join(sample_dir, images[0])
    p2 = os.path.join(sample_dir, images[1])
    
    print(f"  Comparing Face 1: {os.path.basename(p1)}")
    print(f"  Comparing Face 2: {os.path.basename(p2)}")
    print("-" * 60)
    
    # 3. Perform Kinship Verification Inference
    print("[3/3] Running kinship predictions...")
    try:
        # Determine relationship type and build one-hot vector
        rel_type = "father-dau" # matches KFW1 father-dau subfolder
        cat = get_relation_category(rel_type)
        one_hot = [0.0] * 4
        one_hot[cat] = 1.0
        rel_tensor = torch.tensor([one_hot], dtype=torch.float32)
        
        # Extract 512-dim L2-normalized classical embeddings
        emb1 = extractor.extract(p1)
        emb2 = extractor.extract(p2)
        
        # Convert to torch Tensors and add batch dimension (1, 512)
        emb1_tensor = torch.tensor(emb1, dtype=torch.float32).unsqueeze(0)
        emb2_tensor = torch.tensor(emb2, dtype=torch.float32).unsqueeze(0)
        
        # Run Fast Analytical forward pass (passing relation category)
        with torch.no_grad():
            prob_analytical = model(emb1_tensor, emb2_tensor, rel_tensor).item()
            
            # Run actual Qiskit SWAP test circuit on AerSimulator
            x1 = torch.cat([emb1_tensor, rel_tensor], dim=1)
            x2 = torch.cat([emb2_tensor, rel_tensor], dim=1)
            z1_tensor = torch.tanh(model.projection(x1)) * np.pi
            z2_tensor = torch.tanh(model.projection(x2)) * np.pi
            z1 = z1_tensor.squeeze().numpy()
            z2 = z2_tensor.squeeze().numpy()
        
        prob_qiskit = simulate_swap_test(z1, z2, shots=1024)
        
        decision_analytical = "RELATED (KIN)" if prob_analytical >= 0.5 else "NOT RELATED (NON-KIN)"
        decision_qiskit = "RELATED (KIN)" if prob_qiskit >= 0.5 else "NOT RELATED (NON-KIN)"
        
        print("\nPrediction Results:")
        print(f"  * Analytical Quantum Fidelity: {prob_analytical * 100:.2f}% ({decision_analytical})")
        print(f"  * Qiskit Aer SWAP Test (1024 shots): {prob_qiskit * 100:.2f}% ({decision_qiskit})")
        
    except Exception as e:
        print(f"  [ERROR] Inference failed: {e}")
        
    print("=" * 60)

if __name__ == "__main__":
    main()
