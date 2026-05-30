# -*- coding: utf-8 -*-
"""
===============================================================================
  QUANTUM KINSHIP VERIFICATION -- REAL-TIME VISUAL PAIR TESTER & METRICS EXPORTER
===============================================================================

This script:
  1. Loads the trained Hybrid Classical-Quantum Kinship Classifier.
  2. Parses the KinFaceW-I test dataset.
  3. Evaluates a random representative sample of 10 pairs (5 Kin, 5 Non-Kin).
  4. Plots the selected pairs side-by-side and saves a visual summary grid.
  5. Computes overall test metrics (Accuracy, ROC-AUC, Precision, Recall, F1)
     across the entire test set.
  6. Saves all output plots and metrics to results/real_time_test/.

Usage:
  python scripts/test_real_time_pairs.py
"""

import os
import sys
import json
import random
from datetime import datetime
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_curve, auc, precision_recall_fscore_support,
    accuracy_score, confusion_matrix
)

import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.models import FaceFeatureExtractor, HybridKinshipClassifier
from src.quantum_core import simulate_swap_test
from src.data_loaders import (
    load_kinfacew_pairs, cache_face_embeddings, prepare_pair_tensors, get_relation_category
)

# ─── AESTHETICS & COLORS (Slate Premium Dark-Mode) ───────────────────────────
COLOR_PALETTE = {
    'bg_dark':       '#0F172A',   # Slate 900
    'bg_card':       '#1E293B',   # Slate 800
    'text':          '#E2E8F0',   # Slate 200
    'text_muted':    '#94A3B8',   # Slate 400
    'grid':          '#334155',   # Slate 700
    'correct':       '#10B981',   # Emerald 500 (Green)
    'incorrect':     '#EF4444',   # Red 500
    'analytical':    '#38BDF8',   # Sky 400 (Cyan)
    'qiskit':        '#F59E0B',   # Amber 500 (Orange)
}

def setup_plot_style():
    """Configure premium dark-mode matplotlib aesthetics."""
    plt.rcParams.update({
        'figure.facecolor':   COLOR_PALETTE['bg_dark'],
        'axes.facecolor':     COLOR_PALETTE['bg_card'],
        'axes.edgecolor':     COLOR_PALETTE['grid'],
        'axes.labelcolor':    COLOR_PALETTE['text'],
        'axes.titlepad':      14,
        'text.color':         COLOR_PALETTE['text'],
        'xtick.color':        COLOR_PALETTE['text'],
        'ytick.color':        COLOR_PALETTE['text'],
        'grid.color':         COLOR_PALETTE['grid'],
        'grid.alpha':         0.4,
        'grid.linestyle':     '--',
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Segoe UI', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size':          10,
        'legend.facecolor':   COLOR_PALETTE['bg_card'],
        'legend.edgecolor':   COLOR_PALETTE['grid'],
        'legend.fontsize':    9,
        'figure.dpi':         150,
    })

def main():
    setup_plot_style()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = os.path.join(project_root, "weights", "hybrid_kinship.pt")
    kfw1_root = os.path.join(project_root, "KinFaceW-I", "KinFaceW-I")
    output_dir = os.path.join(project_root, "results", "real_time_test")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("      QUANTUM KINSHIP VERIFICATION -- REAL-TIME VISUAL TESTER")
    print("=" * 70)

    # 1. Initialize models & load weights
    print("[1/6] Loading models and checkpoint weights...")
    extractor = FaceFeatureExtractor()
    
    n_qubits = 8
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path)
            # Find qubits count dynamically
            if 'projection.3.bias' in state_dict:
                n_qubits = state_dict['projection.3.bias'].shape[0]
            elif 'projection.4.bias' in state_dict:
                n_qubits = state_dict['projection.4.bias'].shape[0]
            elif 'projection.12.bias' in state_dict:
                n_qubits = state_dict['projection.12.bias'].shape[0]
            elif 'projection.2.bias' in state_dict:
                n_qubits = state_dict['projection.2.bias'].shape[0]
            print(f"  [OK] Detected n_qubits={n_qubits} from checkpoint.")
            model = HybridKinshipClassifier(n_qubits=n_qubits)
            model.load_state_dict(state_dict)
            print(f"  [OK] Loaded weights from: {weights_path}")
        except Exception as e:
            print(f"  [WARN] Failed to load checkpoint: {e}. Using default init.")
            model = HybridKinshipClassifier(n_qubits=n_qubits)
    else:
        print("  [WARN] No checkpoint found. Using default initialization.")
        model = HybridKinshipClassifier(n_qubits=n_qubits)
    
    model.eval()

    # 2. Load KinFaceW-I dataset
    print("\n[2/6] Loading KinFaceW-I test dataset pairs...")
    if not os.path.exists(kfw1_root):
        print(f"  [ERROR] KinFaceW-I dataset directory not found at: {kfw1_root}")
        sys.exit(1)
    
    test_pairs = load_kinfacew_pairs(kfw1_root)
    print(f"  [OK] Successfully loaded {len(test_pairs)} pairs from KinFaceW-I.")

    # 3. Load embedding cache if available (to speed up full test)
    cache_path = os.path.join(project_root, "weights", "embeddings_cache.pkl")
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                raw_cache = pickle_load = json_pickle = pickle_load = pickle_load = pickle_load = pickle_load = None
                import pickle
                raw_cache = pickle.load(f)
            cache = {os.path.normcase(os.path.abspath(k)): v for k, v in raw_cache.items()}
            print(f"  [Cache] Loaded {len(cache)} embeddings from cache.")
        except Exception as e:
            print(f"  [Cache Warning] Failed to load cache: {e}")

    # Ensure all test embeddings are ready
    print("  Ensuring test set embeddings are cached...")
    cache = cache_face_embeddings(test_pairs, extractor, cache_path)

    # 4. Select random subset of 15 pairs (8 Kin, 7 Non-kin) for visual demo
    print("\n[3/6] Selecting 15 random sample pairs (8 Kin, 7 Non-Kin)...")
    kin_pairs = [p for p in test_pairs if p[2] == 1]
    non_kin_pairs = [p for p in test_pairs if p[2] == 0]

    # Deterministic random selection for reproducibility of visuals, yet representative
    rng = random.Random(42)
    selected_kin = rng.sample(kin_pairs, min(8, len(kin_pairs)))
    selected_non_kin = rng.sample(non_kin_pairs, min(7, len(non_kin_pairs)))
    demo_pairs = selected_kin + selected_non_kin
    rng.shuffle(demo_pairs) # Shuffle to mix them up

    # Run predictions on selected pairs
    demo_results = []
    print("\n" + "-" * 105)
    print(f"{'PAIR':<5} | {'RELATION':<10} | {'TRUE':<7} | {'PRED (ANALYTICAL)':<18} | {'PRED (QISKIT)':<14} | {'HYBRID TIME':<12} | {'QISKIT TIME':<12} | {'STATUS':<7}")
    print("-" * 105)
    
    for i, (p1, p2, label, rel) in enumerate(demo_pairs):
        t_start = time.perf_counter()
        
        # Extract features (real-time prediction from raw images)
        emb1 = extractor.extract(p1)
        emb2 = extractor.extract(p2)
        
        t_feat = (time.perf_counter() - t_start) * 1000
        
        # Analytical prediction
        t_anal_start = time.perf_counter()
        emb1_t = torch.tensor(emb1, dtype=torch.float32).unsqueeze(0)
        emb2_t = torch.tensor(emb2, dtype=torch.float32).unsqueeze(0)
        
        cat = get_relation_category(rel, p1)
        one_hot = [0.0] * 4
        one_hot[cat] = 1.0
        rel_t = torch.tensor([one_hot], dtype=torch.float32)
        
        # Inference
        with torch.no_grad():
            prob_anal = model(emb1_t, emb2_t, rel_t).item()
        t_anal_inf = (time.perf_counter() - t_anal_start) * 1000
        
        # Qiskit simulation prediction
        t_qiskit_start = time.perf_counter()
        with torch.no_grad():
            x1 = torch.cat([emb1_t, rel_t], dim=1)
            x2 = torch.cat([emb2_t, rel_t], dim=1)
            z1 = (torch.tanh(model.projection(x1)) * np.pi).squeeze().numpy()
            z2 = (torch.tanh(model.projection(x2)) * np.pi).squeeze().numpy()
            
        prob_qiskit = simulate_swap_test(z1, z2, shots=1024)
        t_qiskit_inf = (time.perf_counter() - t_qiskit_start) * 1000
        
        hybrid_time_ms = t_feat + t_anal_inf
        pred_label = 1 if prob_anal >= 0.5 else 0
        pred_str = "KIN" if pred_label == 1 else "NON-KIN"
        true_str = "KIN" if label == 1 else "NON-KIN"
        status_str = "CORRECT" if pred_label == label else "WRONG"
        
        print(f"#{i+1:<4} | {rel:<10} | {true_str:<7} | {pred_str:<7} ({prob_anal*100:5.1f}%) | {prob_qiskit*100:5.1f}%       | {hybrid_time_ms:6.1f} ms   | {t_qiskit_inf:6.1f} ms   | {status_str:<7}")
        
        demo_results.append({
            'index': i + 1,
            'img1_path': p1,
            'img2_path': p2,
            'relation': rel,
            'true_label': label,
            'true_str': true_str,
            'pred_label': pred_label,
            'pred_str': pred_str,
            'prob_anal': prob_anal,
            'prob_qiskit': prob_qiskit,
            'time_feat': t_feat,
            'time_anal': t_anal_inf,
            'time_qiskit': t_qiskit_inf,
            'hybrid_time': hybrid_time_ms,
            'status': status_str
        })
    print("-" * 105 + "\n")

    # 5. Generate Visual Predictions Plots
    print("[4/6] Generating visual prediction plots (combined grid and 15 individual windows)...")
    
    # 5a. Create combined grid plot (3 rows, 5 columns)
    grid_fig = plt.figure(1000, figsize=(18, 10))
    grid_fig.patch.set_facecolor(COLOR_PALETTE['bg_dark'])
    
    for idx, res in enumerate(demo_results):
        # Plot on the combined grid figure
        plt.figure(grid_fig.number)
        ax = plt.subplot(3, 5, idx + 1)
        ax.set_facecolor(COLOR_PALETTE['bg_card'])
        
        # Load and resize images
        im1 = Image.open(res['img1_path']).convert('RGB').resize((100, 100))
        im2 = Image.open(res['img2_path']).convert('RGB').resize((100, 100))
        
        # Combine side-by-side
        combined_img = Image.new('RGB', (205, 100), (30, 41, 59))
        combined_img.paste(im1, (0, 0))
        combined_img.paste(im2, (105, 0))
        
        ax.imshow(combined_img)
        ax.axis('off')
        
        color = COLOR_PALETTE['correct'] if res['status'] == "CORRECT" else COLOR_PALETTE['incorrect']
        title_text = (
            f"Pair #{res['index']} ({res['relation'].upper()})\n"
            f"True: {res['true_str']} | Pred: {res['pred_str']}\n"
            f"Fidelity: {res['prob_anal']*100:.1f}%\n"
            f"Hybrid: {res['hybrid_time']:.1f}ms | Qiskit: {res['time_qiskit']:.1f}ms\n"
            f"[{res['status']}]"
        )
        ax.set_title(title_text, color=color, fontsize=8, fontweight='bold', pad=8)
        rect = mpatches.Rectangle(
            (-0.02, -0.02), 1.04, 1.04, transform=ax.transAxes,
            fill=False, color=color, linewidth=2, clip_on=False
        )
        ax.add_patch(rect)
        
        # 5b. Create individual figure for this pair
        indiv_fig = plt.figure(idx + 100, figsize=(6, 4.5))
        indiv_fig.patch.set_facecolor(COLOR_PALETTE['bg_dark'])
        ax_ind = indiv_fig.add_subplot(111)
        ax_ind.set_facecolor(COLOR_PALETTE['bg_card'])
        
        # Load larger images for individual window
        im1_l = Image.open(res['img1_path']).convert('RGB').resize((200, 200))
        im2_l = Image.open(res['img2_path']).convert('RGB').resize((200, 200))
        combined_img_l = Image.new('RGB', (410, 200), (30, 41, 59))
        combined_img_l.paste(im1_l, (0, 0))
        combined_img_l.paste(im2_l, (210, 0))
        
        ax_ind.imshow(combined_img_l)
        ax_ind.axis('off')
        
        indiv_title = (
            f"Pair #{res['index']} ({res['relation'].upper()}) - Prediction: {res['status']}\n"
            f"True Status: {res['true_str']} | Predicted Status: {res['pred_str']}\n"
            f"Analytical Fidelity: {res['prob_anal']*100:.2f}% | Qiskit: {res['prob_qiskit']*100:.2f}%\n"
            f"Hybrid Inference Time: {res['hybrid_time']:.1f} ms (Feature Ext: {res['time_feat']:.1f} ms + MLP: {res['time_anal']:.1f} ms)\n"
            f"Qiskit Simulator Time: {res['time_qiskit']:.1f} ms"
        )
        ax_ind.set_title(indiv_title, color=color, fontsize=9, fontweight='bold', pad=10)
        rect_ind = mpatches.Rectangle(
            (-0.01, -0.01), 1.02, 1.02, transform=ax_ind.transAxes,
            fill=False, color=color, linewidth=3, clip_on=False
        )
        ax_ind.add_patch(rect_ind)
        
        # Save individual plot BEFORE show()
        indiv_plot_path = os.path.join(output_dir, f"pair_{res['index']}.png")
        plt.figure(indiv_fig.number)
        plt.tight_layout()
        plt.savefig(indiv_plot_path, facecolor=indiv_fig.get_facecolor(), edgecolor='none')

    # Save the combined grid plot BEFORE show()
    plt.figure(grid_fig.number)
    plt.tight_layout(pad=3.0)
    visual_grid_path = os.path.join(output_dir, "visual_predictions.png")
    plt.savefig(visual_grid_path, facecolor=grid_fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    
    print("Opening 15 visual prediction windows in Python...")
    try:
        # Set window title for grid figure
        if hasattr(grid_fig.canvas, 'manager') and grid_fig.canvas.manager is not None:
            grid_fig.canvas.manager.set_window_title("Real-Time Kinship Predictions - Combined Grid")
            
        # Set window titles for individual figures
        for idx in range(len(demo_results)):
            fig_num = idx + 100
            f = plt.figure(fig_num)
            if hasattr(f.canvas, 'manager') and f.canvas.manager is not None:
                f.canvas.manager.set_window_title(f"Pair #{idx+1} Prediction Details")
                
        # This will open all 16 figures (15 individual + 1 grid) simultaneously in separate desktop windows!
        plt.show()
    except Exception as e:
        print(f"  [INFO] Could not open Python windows (headless environment?): {e}")

    # Close all figures to free memory
    plt.close('all')
    print(f"  [OK] Saved visual comparison grid to: {visual_grid_path}")

    # 6. Evaluate all pairs to compute overall metrics
    print("\n[5/6] Running evaluation on all 1,066 KinFaceW-I test pairs...")
    test_emb1, test_emb2, test_y, test_rel = prepare_pair_tensors(test_pairs, cache)
    
    with torch.no_grad():
        test_preds = model(test_emb1, test_emb2, test_rel).numpy()
    test_y_np = test_y.numpy().squeeze()
    test_preds_squeeze = test_preds.squeeze()
    
    # Calculate metrics
    pred_labels = (test_preds_squeeze >= 0.5).astype(int)
    acc = accuracy_score(test_y_np, pred_labels) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(test_y_np, pred_labels, average='binary')
    
    fpr, tpr, thresholds = roc_curve(test_y_np, test_preds_squeeze)
    roc_auc = auc(fpr, tpr)
    
    tn, fp, fn, tp = confusion_matrix(test_y_np, pred_labels).ravel()
    
    metrics = {
        'total_pairs': len(test_pairs),
        'accuracy': float(acc),
        'roc_auc': float(roc_auc),
        'precision': float(precision * 100),
        'recall': float(recall * 100),
        'f1_score': float(f1 * 100),
        'confusion_matrix': {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        },
        'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  [OK] Saved metrics JSON to: {metrics_path}")

    # Generate Confusion Matrix Heatmap
    print("  Generating Confusion Matrix heatmap...")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor(COLOR_PALETTE['bg_dark'])
    ax.set_facecolor(COLOR_PALETTE['bg_card'])
    
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(cm, cmap='Blues', alpha=0.8)
    
    # Grid labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['NON-KIN', 'KIN'])
    ax.set_yticklabels(['NON-KIN', 'KIN'])
    ax.set_xlabel('Predicted Label', fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontweight='bold', labelpad=10)
    
    # Display counts and percentages inside cells
    for r in range(2):
        for c in range(2):
            count = cm[r, c]
            pct = (count / len(test_y_np)) * 100
            color = 'black' if count > cm.max() / 2 else COLOR_PALETTE['text']
            ax.text(c, r, f"{count}\n({pct:.1f}%)", ha='center', va='center', color=color, fontweight='bold')
            
    ax.set_title("Confusion Matrix (KinFaceW-I Test)", pad=15, fontweight='bold')
    plt.tight_layout()
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"  [OK] Saved Confusion Matrix to: {cm_plot_path}")

    # Generate ROC Curve
    print("  Generating ROC Curve...")
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(COLOR_PALETTE['bg_dark'])
    ax.set_facecolor(COLOR_PALETTE['bg_card'])
    
    ax.plot(fpr, tpr, color=COLOR_PALETTE['analytical'], linewidth=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color=COLOR_PALETTE['text_muted'], linestyle='--', linewidth=1.5, label='Random Guess')
    
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('False Positive Rate (FPR)', labelpad=10)
    ax.set_ylabel('True Positive Rate (TPR)', labelpad=10)
    ax.set_title('Receiver Operating Characteristic (ROC)', pad=15, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    roc_plot_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_plot_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"  [OK] Saved ROC Curve to: {roc_plot_path}")

    # 7. Create Markdown Report
    print("\n[6/6] Generating markdown evaluation report...")
    report_content = f"""# KinFaceW-I Real-Time Evaluation Report

**Generated Date:** {metrics['evaluation_date']}
**Model Configuration:** Hybrid Quantum Kinship Classifier ({n_qubits} qubits, conditioned projection layer)

---

## Overall Performance Metrics

| Metric | Value |
|--------|-------|
| **Total Test Pairs** | {metrics['total_pairs']} |
| **Accuracy** | **{metrics['accuracy']:.2f}%** |
| **ROC-AUC** | **{metrics['roc_auc']:.4f}** |
| **Precision** | **{metrics['precision']:.2f}%** |
| **Recall (TPR)** | **{metrics['recall']:.2f}%** |
| **F1-Score** | **{metrics['f1_score']:.2f}%** |

### Confusion Matrix Breakdown
* **True Positives (TP):** {tp} (Correctly predicted Related)
* **True Negatives (TN):** {tn} (Correctly predicted Unrelated)
* **False Positives (FP):** {fp} (Incorrectly predicted Related)
* **False Negatives (FN):** {fn} (Incorrectly predicted Unrelated)

---

## Sample Pairs Detailed Predictions

Here is the breakdown of the 10 random sample pairs evaluated and visualized in `visual_predictions.png`:

| Pair # | Relation | True Status | Pred Status | Analytical Fidelity | Qiskit Simulation | Status |
|--------|----------|-------------|-------------|---------------------|-------------------|--------|
"""
    for res in demo_results:
        report_content += (
            f"| {res['index']} | {res['relation'].upper()} | {res['true_str']} | "
            f"{res['pred_str']} | {res['prob_anal']*100:.2f}% | {res['prob_qiskit']*100:.2f}% | "
            f"**{res['status']}** |\n"
        )
        
    report_content += """
---

## Visualizations
The following plots are saved in this folder:
- **`visual_predictions.png`**: Side-by-side comparison of facial pairs showing actual vs predicted labels and correctness borders.
- **`confusion_matrix.png`**: Breakdown of model predictions vs actual categories.
- **`roc_curve.png`**: ROC Curve representing true positive vs false positive rate trade-offs.
"""

    report_path = os.path.join(output_dir, "test_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"  [OK] Saved markdown report to: {report_path}")

    print("\n" + "=" * 70)
    print("      EVALUATION COMPLETE - RESULTS SAVED TO results/real_time_test/")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
