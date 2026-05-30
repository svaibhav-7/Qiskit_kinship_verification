# -*- coding: utf-8 -*-
"""
===============================================================================
  QUANTUM KINSHIP VERIFICATION -- PIPELINE TEST & TIMING BENCHMARK
===============================================================================

This script:
  1. Runs end-to-end pipeline tests (data loading, embedding, projection,
     analytical fidelity, Qiskit circuit simulation).
  2. Benchmarks every stage and compares Analytical vs. Qiskit execution time.
  3. Generates 6 publication-quality timing plots in results/.

Usage:
  python scripts/test_pipeline_timing.py
"""

import os
import sys
import time
import json
import pickle
import warnings

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_curve, auc, precision_recall_fscore_support,
    accuracy_score, confusion_matrix
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from src.models import FaceFeatureExtractor, HybridKinshipClassifier
from src.quantum_core import simulate_swap_test, simulate_swap_test_batch, build_swap_test_circuit
from src.data_loaders import (
    load_kinfacew_pairs, load_tskinface_pairs,
    cache_face_embeddings, prepare_pair_tensors, get_relation_category
)


# ─── GLOBAL AESTHETICS ───────────────────────────────────────────────────────
COLOR_PALETTE = {
    'analytical':    '#00D2FF',   # Cyan
    'qiskit':        '#FF6B6B',   # Coral Red
    'embedding':     '#A78BFA',   # Purple
    'projection':    '#34D399',   # Emerald
    'total_hybrid':  '#FBBF24',   # Amber
    'speedup':       '#F472B6',   # Pink
    'bg_dark':       '#0F172A',   # Slate 900
    'bg_card':       '#1E293B',   # Slate 800
    'text':          '#E2E8F0',   # Slate 200
    'grid':          '#334155',   # Slate 700
    'accent':        '#38BDF8',   # Sky 400
}

def setup_plot_style():
    """Configure premium dark-mode matplotlib aesthetics."""
    plt.rcParams.update({
        'figure.facecolor':   COLOR_PALETTE['bg_dark'],
        'axes.facecolor':     COLOR_PALETTE['bg_card'],
        'axes.edgecolor':     COLOR_PALETTE['grid'],
        'axes.labelcolor':    COLOR_PALETTE['text'],
        'axes.titlepad':      18,
        'text.color':         COLOR_PALETTE['text'],
        'xtick.color':        COLOR_PALETTE['text'],
        'ytick.color':        COLOR_PALETTE['text'],
        'grid.color':         COLOR_PALETTE['grid'],
        'grid.alpha':         0.4,
        'grid.linestyle':     '--',
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Segoe UI', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size':          11,
        'legend.facecolor':   COLOR_PALETTE['bg_card'],
        'legend.edgecolor':   COLOR_PALETTE['grid'],
        'legend.fontsize':    10,
        'figure.dpi':         150,
    })


def banner(text, char='=', width=72):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")


# =============================================================================
# TEST 1: Data Loading & Embedding Cache
# =============================================================================

def test_data_loading(project_root):
    """Test that datasets can be loaded and pairs parsed correctly."""
    banner("TEST 1: Dataset Loading & Pair Parsing")
    
    KFW1 = os.path.join(project_root, "KinFaceW-I", "KinFaceW-I")
    KFW2 = os.path.join(project_root, "KinFaceW-II")
    TSKIN = os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data", "TSKinFace_cropped")
    
    results = {}
    
    # Load KinFaceW-I (test set)
    t0 = time.perf_counter()
    if os.path.exists(KFW1):
        test_pairs = load_kinfacew_pairs(KFW1)
        t1 = time.perf_counter()
        results['kfw1_pairs'] = len(test_pairs)
        results['kfw1_load_time'] = t1 - t0
        print(f"  [OK] KinFaceW-I:  {len(test_pairs):,} pairs loaded in {t1-t0:.3f}s")
    else:
        print(f"  [FAIL] KinFaceW-I not found at {KFW1}")
        return None
    
    # Load KinFaceW-II (train set)
    t0 = time.perf_counter()
    train_pairs = []
    if os.path.exists(KFW2):
        kfw2_pairs = load_kinfacew_pairs(KFW2)
        train_pairs.extend(kfw2_pairs)
        t1 = time.perf_counter()
        results['kfw2_pairs'] = len(kfw2_pairs)
        results['kfw2_load_time'] = t1 - t0
        print(f"  [OK] KinFaceW-II: {len(kfw2_pairs):,} pairs loaded in {t1-t0:.3f}s")
    
    # Load TSKinFace (train set)
    t0 = time.perf_counter()
    if os.path.exists(TSKIN):
        ts_pairs = load_tskinface_pairs(TSKIN, max_families=150)
        train_pairs.extend(ts_pairs)
        t1 = time.perf_counter()
        results['tskin_pairs'] = len(ts_pairs)
        results['tskin_load_time'] = t1 - t0
        print(f"  [OK] TSKinFace:   {len(ts_pairs):,} pairs loaded in {t1-t0:.3f}s")
    
    # Validate pair structure
    for p1, p2, label, rel in test_pairs[:5]:
        assert os.path.exists(p1), f"Image not found: {p1}"
        assert os.path.exists(p2), f"Image not found: {p2}"
        assert label in (0, 1), f"Invalid label: {label}"
    print(f"  [OK] Pair structure validation passed (spot-checked 5 pairs)")
    
    # Validate label balance
    kin_count = sum(1 for _, _, l, _ in test_pairs if l == 1)
    nonkin_count = len(test_pairs) - kin_count
    print(f"  [OK] Test set balance: {kin_count} kin / {nonkin_count} non-kin")
    
    results['total_train'] = len(train_pairs)
    results['total_test'] = len(test_pairs)
    
    return test_pairs, train_pairs, results


# =============================================================================
# TEST 2: Embedding Extraction Timing
# =============================================================================

def test_embedding_extraction(project_root, test_pairs, train_pairs):
    """Test embedding cache loading and measure extraction timing."""
    banner("TEST 2: Embedding Cache & Extraction Timing")
    
    cache_path = os.path.join(project_root, "weights", "embeddings_cache.pkl")
    results = {}
    
    t0 = time.perf_counter()
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            raw_cache = pickle.load(f)
        cache = {os.path.normcase(os.path.abspath(k)): v for k, v in raw_cache.items()}
        t1 = time.perf_counter()
        results['cache_load_time'] = t1 - t0
        results['cache_size'] = len(cache)
        print(f"  [OK] Loaded embedding cache: {len(cache):,} entries in {t1-t0:.3f}s")
    else:
        print(f"  [WARN] Cache not found -- initializing extractor to build cache")
        extractor = FaceFeatureExtractor()
        all_pairs = train_pairs + test_pairs
        cache = cache_face_embeddings(all_pairs, extractor, cache_path)
        t1 = time.perf_counter()
        results['cache_build_time'] = t1 - t0
        results['cache_size'] = len(cache)
    
    # Prepare tensors
    t0 = time.perf_counter()
    test_emb1, test_emb2, test_y, test_rel = prepare_pair_tensors(test_pairs, cache)
    train_emb1, train_emb2, train_y, train_rel = prepare_pair_tensors(train_pairs, cache)
    t1 = time.perf_counter()
    results['tensor_prep_time'] = t1 - t0
    
    print(f"  [OK] Tensor preparation: {t1-t0:.4f}s")
    print(f"    Test tensors:  emb1={test_emb1.shape}, labels={test_y.shape}")
    print(f"    Train tensors: emb1={train_emb1.shape}, labels={train_y.shape}")
    
    # Validate embedding dimensions
    assert test_emb1.shape[1] == 512, f"Expected 512-d embeddings, got {test_emb1.shape[1]}"
    assert test_rel.shape[1] == 4, f"Expected 4-d relation one-hot, got {test_rel.shape[1]}"
    print(f"  [OK] Embedding dimension validation: 512-d OK, relation one-hot 4-d OK")
    
    return (test_emb1, test_emb2, test_y, test_rel,
            train_emb1, train_emb2, train_y, train_rel, results)


# =============================================================================
# TEST 3: Model Loading & Forward Pass
# =============================================================================

def test_model_inference(project_root, test_emb1, test_emb2, test_y, test_rel):
    """Test model loading and both analytical / Qiskit inference paths."""
    banner("TEST 3: Model Loading & Inference Paths")
    
    weights_path = os.path.join(project_root, "weights", "hybrid_kinship.pt")
    results = {}
    
    # Load model
    t0 = time.perf_counter()
    n_qubits = 8
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        if 'projection.3.bias' in state_dict:
            n_qubits = state_dict['projection.3.bias'].shape[0]
        model = HybridKinshipClassifier(n_qubits=n_qubits)
        model.load_state_dict(state_dict)
        print(f"  [OK] Model loaded (n_qubits={n_qubits}) from {weights_path}")
    else:
        model = HybridKinshipClassifier(n_qubits=n_qubits)
        print(f"  [WARN] No weights found -- using random initialization (n_qubits={n_qubits})")
    model.eval()
    t1 = time.perf_counter()
    results['model_load_time'] = t1 - t0
    
    # Test analytical forward pass
    t0 = time.perf_counter()
    with torch.no_grad():
        preds_analytical = model(test_emb1, test_emb2, test_rel)
    t1 = time.perf_counter()
    results['analytical_full_time'] = t1 - t0
    results['analytical_per_pair_ms'] = (t1 - t0) / len(test_y) * 1000
    
    preds_np = preds_analytical.numpy().flatten()
    labels_np = test_y.numpy().flatten()
    acc = accuracy_score(labels_np, (preds_np >= 0.5).astype(float)) * 100
    results['analytical_accuracy'] = acc
    
    print(f"  [OK] Analytical forward pass: {len(test_y)} pairs in {t1-t0:.4f}s "
          f"({results['analytical_per_pair_ms']:.3f} ms/pair)")
    print(f"    Accuracy: {acc:.2f}%")
    
    # Test Qiskit forward pass on small subset (full set takes long)
    subset_sizes = [10, 25, 50, 100]
    qiskit_timings = {}
    
    for n in subset_sizes:
        if n > len(test_y):
            break
        t0 = time.perf_counter()
        preds_qiskit = model.forward_qiskit(
            test_emb1[:n], test_emb2[:n], test_rel[:n], shots=1024
        )
        t1 = time.perf_counter()
        qiskit_timings[n] = t1 - t0
        per_pair = (t1 - t0) / n * 1000
        print(f"  [OK] Qiskit SWAP test ({n} pairs, 1024 shots): {t1-t0:.3f}s "
              f"({per_pair:.2f} ms/pair)")
    
    results['qiskit_subset_timings'] = qiskit_timings
    
    # Run Qiskit on full test set for final metrics
    print(f"\n  Running Qiskit on full test set ({len(test_y)} pairs)...")
    t0 = time.perf_counter()
    preds_qiskit_full = model.forward_qiskit(test_emb1, test_emb2, test_rel, shots=1024)
    t1 = time.perf_counter()
    results['qiskit_full_time'] = t1 - t0
    results['qiskit_per_pair_ms'] = (t1 - t0) / len(test_y) * 1000
    
    qpreds_np = preds_qiskit_full.numpy().flatten()
    q_acc = accuracy_score(labels_np, (qpreds_np >= 0.5).astype(float)) * 100
    results['qiskit_accuracy'] = q_acc
    
    print(f"  [OK] Qiskit full test: {t1-t0:.3f}s ({results['qiskit_per_pair_ms']:.2f} ms/pair)")
    print(f"    Qiskit Accuracy: {q_acc:.2f}%")
    
    # Speedup
    speedup = results['qiskit_full_time'] / max(results['analytical_full_time'], 1e-9)
    results['speedup_factor'] = speedup
    print(f"\n  ** SPEEDUP: Analytical is {speedup:.1f}x faster than Qiskit circuit simulation **")
    
    return model, preds_analytical, preds_qiskit_full, results


# =============================================================================
# TEST 4: Per-Stage Timing Benchmark (Varying Batch Sizes)
# =============================================================================

def benchmark_stages(model, test_emb1, test_emb2, test_rel, test_y):
    """Benchmark each pipeline stage at varying batch sizes."""
    banner("TEST 4: Per-Stage Timing Benchmark")
    
    batch_sizes = [1, 5, 10, 25, 50, 100, 200, 500]
    batch_sizes = [b for b in batch_sizes if b <= len(test_y)]
    
    timing_data = {
        'batch_sizes':   [],
        'projection_ms': [],
        'analytical_ms': [],
        'qiskit_ms':     [],
        'total_hybrid_ms': [],
    }
    
    n_qubits = model.n_qubits
    
    for bs in batch_sizes:
        print(f"  Benchmarking batch_size={bs}...")
        
        e1 = test_emb1[:bs]
        e2 = test_emb2[:bs]
        r  = test_rel[:bs]
        
        # Stage A: Classical Projection (512+4 → n_qubits angles)
        times_proj = []
        for _ in range(5):
            t0 = time.perf_counter()
            with torch.no_grad():
                x1 = torch.cat([e1, r], dim=1)
                x2 = torch.cat([e2, r], dim=1)
                z1 = torch.tanh(model.projection(x1)) * np.pi
                z2 = torch.tanh(model.projection(x2)) * np.pi
            t1 = time.perf_counter()
            times_proj.append((t1 - t0) * 1000)
        proj_ms = np.median(times_proj)
        
        # Stage B: Analytical Fidelity (cos²)
        times_anal = []
        for _ in range(5):
            t0 = time.perf_counter()
            with torch.no_grad():
                cos_diff = torch.cos((z1 - z2) / 2.0)
                fidelity = torch.prod(cos_diff ** 2, dim=1, keepdim=True)
            t1 = time.perf_counter()
            times_anal.append((t1 - t0) * 1000)
        anal_ms = np.median(times_anal)
        
        # Stage C: Qiskit SWAP Test Circuit Simulation
        z1_np = z1.cpu().numpy()
        z2_np = z2.cpu().numpy()
        t0 = time.perf_counter()
        _ = simulate_swap_test_batch(z1_np, z2_np, shots=1024)
        t1 = time.perf_counter()
        qiskit_ms = (t1 - t0) * 1000
        
        total_hybrid = proj_ms + anal_ms
        
        timing_data['batch_sizes'].append(bs)
        timing_data['projection_ms'].append(proj_ms)
        timing_data['analytical_ms'].append(anal_ms)
        timing_data['qiskit_ms'].append(qiskit_ms)
        timing_data['total_hybrid_ms'].append(total_hybrid)
        
        print(f"    Projection: {proj_ms:.2f}ms | Analytical: {anal_ms:.3f}ms | "
              f"Qiskit: {qiskit_ms:.1f}ms | Hybrid Total: {total_hybrid:.2f}ms")
    
    return timing_data


# =============================================================================
# TEST 5: Single Pair Circuit Timing (Varying Shots)
# =============================================================================

def benchmark_shots(model, test_emb1, test_emb2, test_rel):
    """Benchmark Qiskit circuit execution time vs. number of shots."""
    banner("TEST 5: Qiskit Shot Count vs. Execution Time")
    
    shot_counts = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    shot_data = {'shots': [], 'time_ms': []}
    
    # Prepare a single pair
    with torch.no_grad():
        x1 = torch.cat([test_emb1[:1], test_rel[:1]], dim=1)
        x2 = torch.cat([test_emb2[:1], test_rel[:1]], dim=1)
        z1 = torch.tanh(model.projection(x1)) * np.pi
        z2 = torch.tanh(model.projection(x2)) * np.pi
    z1_np = z1.squeeze().numpy()
    z2_np = z2.squeeze().numpy()
    
    for shots in shot_counts:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = simulate_swap_test(z1_np, z2_np, shots=shots)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        median_ms = np.median(times)
        shot_data['shots'].append(shots)
        shot_data['time_ms'].append(median_ms)
        print(f"  Shots={shots:5d} -> {median_ms:.2f} ms (median of 3 runs)")
    
    return shot_data


# =============================================================================
# PLOT GENERATION -- 6 PUBLICATION-QUALITY CHARTS
# =============================================================================

def generate_all_plots(timing_data, shot_data, inference_results, results_dir, test_y, preds_analytical, preds_qiskit):
    """Generate 6 premium dark-mode timing & comparison plots."""
    banner("GENERATING PUBLICATION-QUALITY PLOTS")
    setup_plot_style()
    
    bs = timing_data['batch_sizes']
    
    # -- PLOT 1: Analytical vs Qiskit Execution Time (Log Scale) ---------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bs, timing_data['qiskit_ms'], 'o-',
            color=COLOR_PALETTE['qiskit'], linewidth=2.5, markersize=8,
            label='Qiskit AerSimulator (1024 shots)', zorder=5)
    ax.plot(bs, timing_data['total_hybrid_ms'], 's-',
            color=COLOR_PALETTE['analytical'], linewidth=2.5, markersize=8,
            label='Analytical Hybrid (Projection + Fidelity)', zorder=5)
    
    ax.fill_between(bs, timing_data['total_hybrid_ms'], timing_data['qiskit_ms'],
                     alpha=0.15, color=COLOR_PALETTE['speedup'],
                     label='Time Saved (Speedup Region)')
    
    ax.set_yscale('log')
    ax.set_xlabel('Number of Pairs (Batch Size)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Execution Time (ms) - Log Scale', fontweight='bold', fontsize=12)
    ax.set_title('Analytical vs. Qiskit Circuit Simulation Time',
                 fontweight='bold', fontsize=14, color=COLOR_PALETTE['accent'])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    path1 = os.path.join(results_dir, "timing_analytical_vs_qiskit.png")
    plt.savefig(path1)
    plt.close()
    print(f"  [OK] Saved: {path1}")
    
    # -- PLOT 2: Stacked Bar - Per-Stage Breakdown ----------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(bs))
    width = 0.35
    
    # Hybrid stages (stacked)
    bars1 = ax.bar(x - width/2, timing_data['projection_ms'], width,
                   color=COLOR_PALETTE['projection'], label='Classical Projection', edgecolor='none')
    bars2 = ax.bar(x - width/2, timing_data['analytical_ms'], width,
                   bottom=timing_data['projection_ms'],
                   color=COLOR_PALETTE['analytical'], label='Analytical Fidelity', edgecolor='none')
    
    # Qiskit bar
    bars3 = ax.bar(x + width/2, timing_data['qiskit_ms'], width,
                   color=COLOR_PALETTE['qiskit'], label='Qiskit SWAP Circuit', edgecolor='none')
    
    ax.set_xlabel('Number of Pairs', fontweight='bold', fontsize=12)
    ax.set_ylabel('Execution Time (ms)', fontweight='bold', fontsize=12)
    ax.set_title('Per-Stage Time Breakdown: Hybrid vs. Qiskit',
                 fontweight='bold', fontsize=14, color=COLOR_PALETTE['accent'])
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bs])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, axis='y')
    plt.tight_layout()
    path2 = os.path.join(results_dir, "timing_stacked_breakdown.png")
    plt.savefig(path2)
    plt.close()
    print(f"  [OK] Saved: {path2}")
    
    # -- PLOT 3: Speedup Factor Bar Chart -------------------------------------
    speedups = [q / max(h, 0.001) for q, h in
                zip(timing_data['qiskit_ms'], timing_data['total_hybrid_ms'])]
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    gradient_colors = plt.cm.cool(np.linspace(0.2, 0.8, len(bs)))
    bars = ax.bar(range(len(bs)), speedups, color=gradient_colors, edgecolor='none', width=0.65)
    
    for bar, sp in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{sp:.0f}x', ha='center', va='bottom',
                fontweight='bold', fontsize=11, color=COLOR_PALETTE['text'])
    
    ax.set_xlabel('Number of Pairs', fontweight='bold', fontsize=12)
    ax.set_ylabel('Speedup Factor (x faster)', fontweight='bold', fontsize=12)
    ax.set_title('Analytical Speedup over Qiskit Simulation',
                 fontweight='bold', fontsize=14, color=COLOR_PALETTE['accent'])
    ax.set_xticks(range(len(bs)))
    ax.set_xticklabels([str(b) for b in bs])
    ax.grid(True, axis='y')
    ax.axhline(y=1, color=COLOR_PALETTE['qiskit'], linestyle='--', linewidth=1.5, alpha=0.7,
               label='Parity (1x)')
    ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    path3 = os.path.join(results_dir, "timing_speedup_factor.png")
    plt.savefig(path3)
    plt.close()
    print(f"  [OK] Saved: {path3}")
    
    # -- PLOT 4: Shot Count vs. Circuit Execution Time ------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(shot_data['shots'], shot_data['time_ms'], 'D-',
            color=COLOR_PALETTE['qiskit'], linewidth=2.5, markersize=8)
    
    # Add analytical baseline
    # Analytical time for 1 pair is essentially near 0
    anal_time_1 = inference_results.get('analytical_per_pair_ms', 0.01)
    ax.axhline(y=anal_time_1, color=COLOR_PALETTE['analytical'],
               linestyle='--', linewidth=2, alpha=0.9,
               label=f'Analytical (constant ≈ {anal_time_1:.3f} ms/pair)')
    
    ax.fill_between(shot_data['shots'], anal_time_1, shot_data['time_ms'],
                     alpha=0.12, color=COLOR_PALETTE['speedup'])
    
    ax.set_xlabel('Number of Shots', fontweight='bold', fontsize=12)
    ax.set_ylabel('Execution Time per Pair (ms)', fontweight='bold', fontsize=12)
    ax.set_title('Qiskit Circuit Time vs. Shot Count (Single Pair)',
                 fontweight='bold', fontsize=14, color=COLOR_PALETTE['accent'])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True)
    plt.tight_layout()
    path4 = os.path.join(results_dir, "timing_shots_vs_time.png")
    plt.savefig(path4)
    plt.close()
    print(f"  [OK] Saved: {path4}")
    
    # -- PLOT 5: Per-Pair Time Comparison (Analytical vs Qiskit) ---------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    per_pair_anal = [t / b for t, b in zip(timing_data['total_hybrid_ms'], bs)]
    per_pair_qiskit = [t / b for t, b in zip(timing_data['qiskit_ms'], bs)]
    
    ax.plot(bs, per_pair_qiskit, 'o-',
            color=COLOR_PALETTE['qiskit'], linewidth=2.5, markersize=8,
            label='Qiskit (ms/pair)')
    ax.plot(bs, per_pair_anal, 's-',
            color=COLOR_PALETTE['analytical'], linewidth=2.5, markersize=8,
            label='Analytical Hybrid (ms/pair)')
    
    ax.set_xlabel('Batch Size', fontweight='bold', fontsize=12)
    ax.set_ylabel('Time per Pair (ms)', fontweight='bold', fontsize=12)
    ax.set_title('Per-Pair Inference Latency: Hybrid vs. Qiskit',
                 fontweight='bold', fontsize=14, color=COLOR_PALETTE['accent'])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    path5 = os.path.join(results_dir, "timing_per_pair_latency.png")
    plt.savefig(path5)
    plt.close()
    print(f"  [OK] Saved: {path5}")
    
    # -- PLOT 6: Combined Dashboard - 4-Panel Summary -------------------------
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Quantum Kinship Verification — Pipeline Timing Dashboard',
                 fontsize=16, fontweight='bold', color=COLOR_PALETTE['accent'], y=0.97)
    
    # Panel A: Log-scale comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(bs, timing_data['qiskit_ms'], 'o-',
             color=COLOR_PALETTE['qiskit'], linewidth=2, markersize=6, label='Qiskit')
    ax1.plot(bs, timing_data['total_hybrid_ms'], 's-',
             color=COLOR_PALETTE['analytical'], linewidth=2, markersize=6, label='Analytical')
    ax1.fill_between(bs, timing_data['total_hybrid_ms'], timing_data['qiskit_ms'],
                      alpha=0.12, color=COLOR_PALETTE['speedup'])
    ax1.set_yscale('log')
    ax1.set_xlabel('Batch Size', fontsize=10)
    ax1.set_ylabel('Time (ms) — Log', fontsize=10)
    ax1.set_title('A. Execution Time Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True)
    
    # Panel B: Speedup bars
    ax2 = fig.add_subplot(gs[0, 1])
    gradient_colors2 = plt.cm.cool(np.linspace(0.2, 0.8, len(bs)))
    ax2.bar(range(len(bs)), speedups, color=gradient_colors2, edgecolor='none', width=0.6)
    for i, sp in enumerate(speedups):
        ax2.text(i, sp + 0.3, f'{sp:.0f}×', ha='center', fontweight='bold', fontsize=9,
                 color=COLOR_PALETTE['text'])
    ax2.set_xticks(range(len(bs)))
    ax2.set_xticklabels([str(b) for b in bs], fontsize=9)
    ax2.set_xlabel('Batch Size', fontsize=10)
    ax2.set_ylabel('Speedup (×)', fontsize=10)
    ax2.set_title('B. Analytical Speedup Factor', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y')
    
    # Panel C: Shot count vs time
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(shot_data['shots'], shot_data['time_ms'], 'D-',
             color=COLOR_PALETTE['qiskit'], linewidth=2, markersize=6)
    ax3.axhline(y=anal_time_1, color=COLOR_PALETTE['analytical'],
                linestyle='--', linewidth=1.5, alpha=0.9, label=f'Analytical ≈ {anal_time_1:.3f}ms')
    ax3.set_xlabel('Shots', fontsize=10)
    ax3.set_ylabel('Time (ms)', fontsize=10)
    ax3.set_title('C. Circuit Time vs. Shot Count', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True)
    
    # Panel D: Accuracy comparison + full pipeline summary
    ax4 = fig.add_subplot(gs[1, 1])
    labels_np = test_y.numpy().flatten()
    
    anal_preds = preds_analytical.numpy().flatten()
    qiskit_preds = preds_qiskit.numpy().flatten()
    
    # ROC curves
    fpr_a, tpr_a, _ = roc_curve(labels_np, anal_preds)
    auc_a = auc(fpr_a, tpr_a)
    fpr_q, tpr_q, _ = roc_curve(labels_np, qiskit_preds)
    auc_q = auc(fpr_q, tpr_q)
    
    ax4.plot(fpr_a, tpr_a, color=COLOR_PALETTE['analytical'], linewidth=2,
             label=f'Analytical (AUC={auc_a:.4f})')
    ax4.plot(fpr_q, tpr_q, color=COLOR_PALETTE['qiskit'], linewidth=2, linestyle='--',
             label=f'Qiskit (AUC={auc_q:.4f})')
    ax4.plot([0,1], [0,1], color=COLOR_PALETTE['grid'], linestyle=':', linewidth=1)
    ax4.set_xlabel('FPR', fontsize=10)
    ax4.set_ylabel('TPR', fontsize=10)
    ax4.set_title('D. ROC Curves (Accuracy Equivalence)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='lower right')
    ax4.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path6 = os.path.join(results_dir, "timing_dashboard.png")
    plt.savefig(path6)
    plt.close()
    print(f"  [OK] Saved: {path6}")
    
    return [path1, path2, path3, path4, path5, path6]


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    banner("QUANTUM KINSHIP VERIFICATION -- PIPELINE TEST & TIMING BENCHMARK", '#')
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "results", "timing_benchmarks")
    os.makedirs(results_dir, exist_ok=True)
    
    total_start = time.perf_counter()
    all_results = {}
    
    # TEST 1: Data Loading
    data_result = test_data_loading(project_root)
    if data_result is None:
        print("FATAL: Cannot proceed without test data.")
        sys.exit(1)
    test_pairs, train_pairs, load_results = data_result
    all_results['data_loading'] = load_results
    
    # TEST 2: Embedding Extraction
    (test_emb1, test_emb2, test_y, test_rel,
     train_emb1, train_emb2, train_y, train_rel, emb_results) = \
        test_embedding_extraction(project_root, test_pairs, train_pairs)
    all_results['embeddings'] = emb_results
    
    # TEST 3: Model Inference
    model, preds_analytical, preds_qiskit, inf_results = \
        test_model_inference(project_root, test_emb1, test_emb2, test_y, test_rel)
    all_results['inference'] = inf_results
    
    # TEST 4: Per-Stage Benchmark
    timing_data = benchmark_stages(model, test_emb1, test_emb2, test_rel, test_y)
    all_results['stage_timing'] = timing_data
    
    # TEST 5: Shot Count Benchmark
    shot_data = benchmark_shots(model, test_emb1, test_emb2, test_rel)
    all_results['shot_timing'] = shot_data
    
    # GENERATE PLOTS
    plot_paths = generate_all_plots(
        timing_data, shot_data, inf_results, results_dir,
        test_y, preds_analytical, preds_qiskit
    )
    
    total_time = time.perf_counter() - total_start
    
    # -- FINAL SUMMARY ---------------------------------------------------------
    banner("FINAL TEST SUMMARY")
    
    print(f"  Total benchmark time:    {total_time:.1f}s")
    print(f"  Datasets loaded:         {all_results['data_loading'].get('total_test', 0)} test / "
          f"{all_results['data_loading'].get('total_train', 0)} train pairs")
    print(f"  Embedding cache:         {all_results['embeddings'].get('cache_size', 0)} entries")
    print(f"  Analytical accuracy:     {inf_results['analytical_accuracy']:.2f}%")
    print(f"  Qiskit accuracy:         {inf_results['qiskit_accuracy']:.2f}%")
    print(f"  Analytical time (full):  {inf_results['analytical_full_time']*1000:.2f} ms "
          f"({inf_results['analytical_per_pair_ms']:.3f} ms/pair)")
    print(f"  Qiskit time (full):      {inf_results['qiskit_full_time']*1000:.1f} ms "
          f"({inf_results['qiskit_per_pair_ms']:.2f} ms/pair)")
    print(f"  SPEEDUP FACTOR:          {inf_results['speedup_factor']:.0f}x")
    print(f"\n  Plots generated ({len(plot_paths)}):")
    for p in plot_paths:
        print(f"    -> {os.path.basename(p)}")
    
    # Save raw timing data as JSON
    json_path = os.path.join(results_dir, "pipeline_timing_results.json")
    
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    serializable = json.loads(json.dumps(all_results, default=convert))
    with open(json_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Raw timing data saved to: {json_path}")
    
    banner("ALL TESTS PASSED", '#')


if __name__ == "__main__":
    main()
