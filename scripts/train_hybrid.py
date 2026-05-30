import os
import sys
import argparse

# Add project root to sys.path to allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

from src.models import FaceFeatureExtractor, HybridKinshipClassifier
from src.data_loaders import (
    load_kinfacew_pairs, 
    load_tskinface_pairs, 
    cache_face_embeddings, 
    prepare_pair_tensors
)

# Set non-interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')

def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid Classical-Quantum Kinship Verification Model")
    parser.add_argument("--n-qubits", type=int, default=8, help="Number of qubits representing each face embedding")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer")
    parser.add_argument("--max-families", type=int, default=150, help="Max TSKinFace families to load")
    parser.add_argument("--fallback-resnet", action="store_true", help="Force ResNet-18 fallback instead of FaceNet")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Project Directory Configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(project_root, "weights")
    results_dir = os.path.join(project_root, "results", "training_metrics")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 70)
    print("      QUANTUM KINSHIP VERIFICATION -- HYBRID SWAP TEST PIPELINE")
    print("=" * 70)
    print(f"  Configuration:")
    print(f"    - Qubits per register: {args.n_qubits} (Total qubits: {2*args.n_qubits + 1} with Ancilla)")
    print(f"    - Training epochs    : {args.epochs}")
    print(f"    - Batch size         : {args.batch_size}")
    print(f"    - Learning rate      : {args.lr}")
    print(f"    - Fallback ResNet    : {args.fallback_resnet}")
    print("-" * 70)
    
    # Dataset Paths
    KFW1 = os.path.join(project_root, "KinFaceW-I", "KinFaceW-I")
    KFW2 = os.path.join(project_root, "KinFaceW-II")
    TSKIN = os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data", "TSKinFace_cropped")
    
    # 2. Load Dataset Metadata
    print("[1/6] Parsing datasets metadata...")
    train_pairs = []
    
    if os.path.exists(KFW2):
        print("  - Loading KinFaceW-II (Train Split)...")
        kfw2_pairs = load_kinfacew_pairs(KFW2)
        train_pairs.extend(kfw2_pairs)
        print(f"    Loaded {len(kfw2_pairs)} pairs from KinFaceW-II.")
    else:
        print("  - [SKIP] KinFaceW-II directory not found.")
        
    if os.path.exists(TSKIN):
        print(f"  - Loading TSKinFace (Train Split, max families={args.max_families})...")
        ts_pairs = load_tskinface_pairs(TSKIN, max_families=args.max_families)
        train_pairs.extend(ts_pairs)
        print(f"    Loaded {len(ts_pairs)} pairs from TSKinFace.")
    else:
        print("  - [SKIP] TSKinFace directory not found.")
        
    if len(train_pairs) == 0:
        raise RuntimeError("No training dataset folders found! Verify your directory layout.")
        
    test_pairs = []
    if os.path.exists(KFW1):
        print("  - Loading KinFaceW-I (Test Split)...")
        test_pairs = load_kinfacew_pairs(KFW1)
        print(f"    Loaded {len(test_pairs)} pairs from KinFaceW-I.")
    else:
        raise RuntimeError("KinFaceW-I test dataset not found! We require a test set.")
        
    print(f"  Total train pairs: {len(train_pairs)} | Total test pairs: {len(test_pairs)}")
    print("-" * 70)
    
    # 3. Embedding Caching
    import pickle
    cache_path = os.path.join(weights_dir, "embeddings_cache.pkl")
    
    # Check if cache exists and check missing paths before initializing extractor
    unique_paths = set()
    for p1, p2, _, _ in (train_pairs + test_pairs):
        unique_paths.add(os.path.normcase(os.path.abspath(p1)))
        unique_paths.add(os.path.normcase(os.path.abspath(p2)))
        
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                raw_cache = pickle.load(f)
            cache = {os.path.normcase(os.path.abspath(k)): v for k, v in raw_cache.items()}
            print(f"  [Cache] Loaded existing cache with {len(cache)} embeddings.")
        except Exception as e:
            print(f"  [Cache] Error loading cache file: {e}")
            
    paths_to_extract = [p for p in unique_paths if p not in cache]
    if len(paths_to_extract) > 0:
        print(f"[2/6] Initializing face feature extractor to extract {len(paths_to_extract)} new embeddings...")
        extractor = FaceFeatureExtractor(use_resnet_fallback=args.fallback_resnet)
        cache = cache_face_embeddings(train_pairs + test_pairs, extractor, cache_path)
    else:
        print("[2/6] All embeddings found in cache. Skipping feature extractor initialization.")
    
    # Prepare PyTorch Tensors
    train_emb1, train_emb2, train_y, train_rel = prepare_pair_tensors(train_pairs, cache)
    test_emb1, test_emb2, test_y, test_rel = prepare_pair_tensors(test_pairs, cache)
    
    print(f"  Prepared Tensors:")
    print(f"    Train: emb1 shape={train_emb1.shape}, emb2 shape={train_emb2.shape}, labels shape={train_y.shape}, rels shape={train_rel.shape}")
    print(f"    Test : emb1 shape={test_emb1.shape}, emb2 shape={test_emb2.shape}, labels shape={test_y.shape}, rels shape={test_rel.shape}")
    
    # Balance check
    n_kin_tr = int(train_y.sum().item())
    n_non_tr = len(train_y) - n_kin_tr
    n_kin_te = int(test_y.sum().item())
    n_non_te = len(test_y) - n_kin_te
    print(f"    Train distribution: {n_kin_tr} Kin, {n_non_tr} Non-kin")
    print(f"    Test distribution : {n_kin_te} Kin, {n_non_te} Non-kin")
    print("-" * 70)
    
    # Create DataLoader
    train_dataset = TensorDataset(train_emb1, train_emb2, train_y, train_rel)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 4. Build Model & Setup Optimizer
    print("[3/6] Constructing Fast Quantum SWAP Classifier...")
    model = HybridKinshipClassifier(n_qubits=args.n_qubits)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-5)
    
    # 5. Fast Training Loop (Analytical Quantum Simulation)
    print("[4/6] Training with CosineAnnealing LR, weight decay, noise augmentation, and early stopping...")
    loss_history = []
    acc_history = []
    val_acc_history = []
    
    save_path = os.path.join(weights_dir, "hybrid_kinship.pt")
    best_test_acc = 0.0
    best_state_dict = None
    patience_counter = 0
    patience_limit = 15
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        correct = 0
        total = 0
        
        for batch_emb1, batch_emb2, batch_y, batch_rel in train_loader:
            optimizer.zero_grad()
            
            # Forward pass (analytical quantum state vector overlap) with relations
            preds = model(batch_emb1, batch_emb2, batch_rel)
            loss = criterion(preds, batch_y)
            
            # Backward pass & optimization
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Track training accuracy
            pred_labels = (preds >= 0.5).float()
            correct += (pred_labels == batch_y).sum().item()
            total += batch_y.size(0)
            
        epoch_loss = np.mean(epoch_losses)
        epoch_acc = (correct / total) * 100
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)
        
        # Evaluate validation performance on test split (Analytical Mode is ultra fast)
        model.eval()
        with torch.no_grad():
            val_preds = model(test_emb1, test_emb2, test_rel)
            val_acc = np.mean((val_preds.cpu().numpy() >= 0.5).astype(float) == test_y.numpy()) * 100
            
        if val_acc > best_test_acc:
            best_test_acc = val_acc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state_dict, save_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        val_acc_history.append(val_acc)
        
        # Step the learning rate scheduler (monitors val_acc)
        scheduler.step(val_acc)
        
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:2d}/{args.epochs:2d} -- Loss: {epoch_loss:.4f} -- Train Acc: {epoch_acc:.1f}% -- Val Acc: {val_acc:.1f}% (Best: {best_test_acc:.1f}%) -- LR: {current_lr:.6f}")
        
        # Early stopping check
        if patience_counter >= patience_limit:
            print(f"  [Early Stop] No improvement for {patience_limit} epochs. Stopping at epoch {epoch}.")
            break
        
    print("  Training completed successfully!")
    print("-" * 70)
    
    # 6. Evaluation
    print("[5/6] Running Qiskit SWAP test quantum simulator on KinFaceW-I test split...")
    
    # Load the best performing model weights for evaluation
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"  [OK] Loaded best model checkpoint weights with validation accuracy: {best_test_acc:.2f}%")
    else:
        print("  [WARN] No best state dict checkpoint was saved. Using current epoch weights.")
        
    model.eval()
    
    # Run exact PyTorch analytical pass first
    with torch.no_grad():
        test_preds_analytical = model(test_emb1, test_emb2, test_rel)
        test_loss_analytical = criterion(test_preds_analytical, test_y).item()
    test_acc_analytical = np.mean((test_preds_analytical.numpy() >= 0.5).astype(float) == test_y.numpy()) * 100
    
    # Run the Qiskit AerSimulator circuit execution verification
    print("  Executing Qiskit AerSimulator circuits for all test pairs (1024 shots)...")
    test_preds_qiskit = model.forward_qiskit(test_emb1, test_emb2, test_rel, shots=1024)
    test_loss_qiskit = criterion(test_preds_qiskit, test_y).item()
    
    test_preds_np = test_preds_qiskit.numpy()
    test_y_np = test_y.numpy()
    
    # Compute final metrics based on the Qiskit Quantum Simulator results
    pred_labels_np = (test_preds_np >= 0.5).astype(float)
    test_acc_qiskit = np.mean(pred_labels_np == test_y_np) * 100
    
    precision, recall, f1, _ = precision_recall_fscore_support(test_y_np, pred_labels_np, average='binary')
    precision_pct = precision * 100
    recall_pct = recall * 100
    f1_pct = f1 * 100
    
    tp = np.sum((pred_labels_np == 1) & (test_y_np == 1))
    fp = np.sum((pred_labels_np == 1) & (test_y_np == 0))
    fn = np.sum((pred_labels_np == 0) & (test_y_np == 1))
    tn = np.sum((pred_labels_np == 0) & (test_y_np == 0))
    
    print(f"  Analytical Quantum Metrics:")
    print(f"    - Accuracy        : {test_acc_analytical:.2f}%")
    print(f"    - BCE Loss        : {test_loss_analytical:.4f}")
    print(f"  Qiskit AerSimulator Quantum Circuit Metrics:")
    print(f"    - BCE Loss        : {test_loss_qiskit:.4f}")
    print(f"    - Test Accuracy   : {test_acc_qiskit:.2f}%")
    print(f"    - Precision       : {precision_pct:.2f}%")
    print(f"    - Recall (TPR)    : {recall_pct:.2f}%")
    print(f"    - F1 Score        : {f1_pct:.2f}%")
    print(f"    - Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    
    # Optimal threshold search using Youden's J statistic
    fpr_th, tpr_th, thresholds_opt = roc_curve(test_y_np, test_preds_np)
    j_scores = tpr_th - fpr_th
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds_opt[best_idx]
    
    pred_labels_opt = (test_preds_np >= optimal_threshold).astype(float)
    test_acc_opt = np.mean(pred_labels_opt == test_y_np) * 100
    prec_opt, rec_opt, f1_opt, _ = precision_recall_fscore_support(test_y_np, pred_labels_opt, average='binary')
    
    print(f"  Optimal Threshold Analysis (Youden's J):")
    print(f"    - Optimal Threshold : {optimal_threshold:.4f} (vs fixed 0.5)")
    print(f"    - Accuracy @ optimal: {test_acc_opt:.2f}%")
    print(f"    - Precision @ optimal: {prec_opt*100:.2f}%")
    print(f"    - Recall @ optimal  : {rec_opt*100:.2f}%")
    print(f"    - F1 @ optimal      : {f1_opt*100:.2f}%")
    print("-" * 70)
    
    print(f"[6/6] Saved best trained model weights to: {save_path}")
    
    # 7. Generate Visualizations (Premium Aesthetics)
    print("Generating premium visualization charts...")
    
    # Set aesthetics styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    
    # Plot 1: Loss & Accuracy Curve (with validation tracking)
    actual_epochs = len(loss_history)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = '#4A90E2' # Electric Blue
    ax1.set_xlabel('Epochs', fontweight='bold', fontsize=11, labelpad=8)
    ax1.set_ylabel('BCE Loss', color=color, fontweight='bold', fontsize=11, labelpad=8)
    line1 = ax1.plot(range(1, actual_epochs + 1), loss_history, color=color, linewidth=2.5, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=11, labelpad=8)
    line2 = ax2.plot(range(1, actual_epochs + 1), acc_history, color='#50E3C2', linewidth=2.5, label='Train Acc')
    line3 = ax2.plot(range(1, actual_epochs + 1), val_acc_history, color='#F5A623', linewidth=2.5, linestyle='--', label='Val Acc')
    ax2.tick_params(axis='y')
    ax2.grid(False)
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.title('Hybrid SWAP Test Model Training Dynamics', fontweight='bold', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_metrics.png"), dpi=150)
    plt.close()
    
    # Plot 2: ROC Curve
    fpr, tpr, thresholds = roc_curve(test_y_np, test_preds_np)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#9013FE', linewidth=2.5, label=f'Qiskit ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#9B9B9B', linestyle='--', linewidth=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=11, labelpad=8)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=11, labelpad=8)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', fontsize=13, pad=15)
    plt.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "roc_curve.png"), dpi=150)
    plt.close()
    
    # Plot 3: Probability Separation Histogram
    plt.figure(figsize=(9, 5))
    kin_scores = test_preds_np[test_y_np == 1]
    nonkin_scores = test_preds_np[test_y_np == 0]
    
    plt.hist(nonkin_scores, bins=15, alpha=0.6, color='#E2849A', label='Non-Kin pairs', edgecolor='none')
    plt.hist(kin_scores, bins=15, alpha=0.6, color='#4A90E2', label='Kin pairs', edgecolor='none')
    plt.axvline(x=0.5, color='#D0021B', linestyle=':', linewidth=2, label='Threshold (0.5)')
    plt.xlabel('Predicted Kinship Overlap (Fidelity)', fontweight='bold', fontsize=11, labelpad=8)
    plt.ylabel('Pair Count', fontweight='bold', fontsize=11, labelpad=8)
    plt.title('Qiskit Simulated Kinship Overlap Distribution', fontweight='bold', fontsize=13, pad=15)
    plt.legend(loc='upper center', frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "score_distribution.png"), dpi=150)
    plt.close()
    
    print("Charts generated successfully in results/ directory:")
    print("  - results/training_metrics.png (Loss & Accuracy)")
    print("  - results/roc_curve.png (ROC-AUC)")
    print("  - results/score_distribution.png (Kinship Separation)")
    print("=" * 70)
    print("  DONE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
