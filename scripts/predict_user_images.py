import os
import sys
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import FaceFeatureExtractor, HybridKinshipClassifier
from src.data_loaders import get_relation_category

def setup_plot_style():
    """Configure premium dark-mode matplotlib aesthetics."""
    plt.rcParams.update({
        'figure.facecolor':   '#0F172A',
        'axes.facecolor':     '#1E293B',
        'axes.edgecolor':     '#334155',
        'axes.labelcolor':    '#E2E8F0',
        'text.color':         '#E2E8F0',
        'xtick.color':        '#E2E8F0',
        'ytick.color':        '#E2E8F0',
        'font.family':        'sans-serif',
        'figure.dpi':         150,
    })

def main():
    parser = argparse.ArgumentParser(description="Predict kinship between two images.")
    parser.add_argument("image1", help="Path to the first image")
    parser.add_argument("image2", help="Path to the second image")
    parser.add_argument("--relation", type=str, choices=['fs', 'fd', 'ms', 'md', 'unknown'], default='unknown',
                        help="Relationship type: fs (father-son), fd (father-daughter), ms (mother-son), md (mother-daughter)")
    parser.add_argument("--weights", type=str, default="weights/hybrid_kinship.pt",
                        help="Path to model weights")
    parser.add_argument("--qiskit", action="store_true", help="Also run Qiskit simulation (slower)")
    parser.add_argument("--no-plot", action="store_true", help="Do not show the plot window")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = os.path.join(project_root, args.weights) if not os.path.isabs(args.weights) else args.weights

    if not os.path.exists(args.image1):
        print(f"Error: Image 1 not found at {args.image1}")
        return
    if not os.path.exists(args.image2):
        print(f"Error: Image 2 not found at {args.image2}")
        return

    print("=" * 60)
    print("   QUANTUM KINSHIP VERIFICATION -- USER IMAGE PREDICTOR")
    print("=" * 60)

    # 1. Initialize models & load weights
    print("[1/3] Loading models and checkpoint weights...")
    extractor = FaceFeatureExtractor()
    
    n_qubits = 8
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            # Find qubits count dynamically
            if 'projection.3.bias' in state_dict:
                n_qubits = state_dict['projection.3.bias'].shape[0]
            elif 'projection.4.bias' in state_dict:
                n_qubits = state_dict['projection.4.bias'].shape[0]
            elif 'projection.12.bias' in state_dict:
                n_qubits = state_dict['projection.12.bias'].shape[0]
            elif 'projection.2.bias' in state_dict:
                n_qubits = state_dict['projection.2.bias'].shape[0]
            model = HybridKinshipClassifier(n_qubits=n_qubits)
            model.load_state_dict(state_dict)
            print(f"  [OK] Loaded weights from: {weights_path}")
        except Exception as e:
            print(f"  [WARN] Failed to load checkpoint: {e}. Using default init.")
            model = HybridKinshipClassifier(n_qubits=n_qubits)
    else:
        print(f"  [WARN] No checkpoint found at {weights_path}. Using default initialization.")
        model = HybridKinshipClassifier(n_qubits=n_qubits)
    
    model.eval()

    # 2. Extract features
    print("\n[2/3] Extracting facial features...")
    t_start = time.perf_counter()
    emb1 = extractor.extract(args.image1)
    emb2 = extractor.extract(args.image2)
    t_feat = (time.perf_counter() - t_start) * 1000
    print(f"  [OK] Feature extraction took {t_feat:.1f} ms")

    # 3. Predict Kinship
    print("\n[3/3] Predicting kinship...")
    
    emb1_t = torch.tensor(emb1, dtype=torch.float32).unsqueeze(0)
    emb2_t = torch.tensor(emb2, dtype=torch.float32).unsqueeze(0)
    
    one_hot = [0.0] * 4
    if args.relation != 'unknown':
        cat = get_relation_category(args.relation, args.image1)
        one_hot[cat] = 1.0
    else:
        # Uniform distribution if relation is unknown
        one_hot = [0.25, 0.25, 0.25, 0.25]
        
    rel_t = torch.tensor([one_hot], dtype=torch.float32)
    
    # Analytical Prediction
    t_anal_start = time.perf_counter()
    with torch.no_grad():
        prob_anal = model(emb1_t, emb2_t, rel_t).item()
    t_anal_inf = (time.perf_counter() - t_anal_start) * 1000
    
    pred_anal = 1 if prob_anal > 0.5 else 0
    pred_str = "KIN" if pred_anal == 1 else "NON-KIN"
    
    print("\n" + "-" * 50)
    print("RESULTS:")
    print("-" * 50)
    print(f"Prediction:          {pred_str}")
    print(f"Confidence (Hybrid): {prob_anal * 100:.2f}%")
    print(f"Inference Time:      {t_anal_inf:.2f} ms")
    
    if args.qiskit:
        from src.quantum_core import simulate_swap_test
        t_qiskit_start = time.perf_counter()
        with torch.no_grad():
            x1 = torch.cat([emb1_t, rel_t], dim=1)
            x2 = torch.cat([emb2_t, rel_t], dim=1)
            
            proj1 = model.projection(x1).squeeze(0).numpy()
            proj2 = model.projection(x2).squeeze(0).numpy()
            
            # Re-scale to [0, pi/2] roughly for embedding
            proj1_scaled = np.clip(proj1, -1, 1) * (np.pi / 2)
            proj2_scaled = np.clip(proj2, -1, 1) * (np.pi / 2)
            
            prob_qiskit = simulate_swap_test(proj1_scaled, proj2_scaled, shots=1024)
        t_qiskit_inf = (time.perf_counter() - t_qiskit_start) * 1000
        print(f"Confidence (Qiskit): {prob_qiskit * 100:.2f}%")
        print(f"Qiskit Sim Time:     {t_qiskit_inf:.2f} ms")
    print("-" * 50)

    # 4. Display result
    if not args.no_plot:
        setup_plot_style()
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        try:
            img1 = Image.open(args.image1).convert('RGB')
            img2 = Image.open(args.image2).convert('RGB')
            axes[0].imshow(img1)
            axes[1].imshow(img2)
        except Exception as e:
            print(f"Could not load images for plotting: {e}")
            return
            
        axes[0].axis('off')
        axes[1].axis('off')
        axes[0].set_title("Image 1")
        axes[1].set_title("Image 2")
        
        color = '#10B981' if pred_anal == 1 else '#EF4444' # Green for Kin, Red for Non-kin
        fig.suptitle(f"Prediction: {pred_str} ({prob_anal*100:.1f}%)", 
                     color=color, fontsize=16, fontweight='bold', y=0.95)
        
        # Add a border around the whole figure based on prediction
        fig.patch.set_edgecolor(color)
        fig.patch.set_linewidth(4)
        
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        print("\nOpening plot window. Close the window to exit the script.")
        plt.show()

if __name__ == "__main__":
    main()
