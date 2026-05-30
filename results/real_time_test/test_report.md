# KinFaceW-I Real-Time Evaluation Report

**Generated Date:** 2026-05-30 13:25:42
**Model Configuration:** Hybrid Quantum Kinship Classifier (8 qubits, conditioned projection layer)

---

## Overall Performance Metrics

| Metric | Value |
|--------|-------|
| **Total Test Pairs** | 1066 |
| **Accuracy** | **68.11%** |
| **ROC-AUC** | **0.7280** |
| **Precision** | **67.51%** |
| **Recall (TPR)** | **69.79%** |
| **F1-Score** | **68.63%** |

### Confusion Matrix Breakdown
* **True Positives (TP):** 372 (Correctly predicted Related)
* **True Negatives (TN):** 354 (Correctly predicted Unrelated)
* **False Positives (FP):** 179 (Incorrectly predicted Related)
* **False Negatives (FN):** 161 (Incorrectly predicted Unrelated)

---

## Sample Pairs Detailed Predictions

Here is the breakdown of the 10 random sample pairs evaluated and visualized in `visual_predictions.png`:

| Pair # | Relation | True Status | Pred Status | Analytical Fidelity | Qiskit Simulation | Status |
|--------|----------|-------------|-------------|---------------------|-------------------|--------|
| 1 | FD | NON-KIN | NON-KIN | 19.01% | 12.30% | **CORRECT** |
| 2 | FD | NON-KIN | NON-KIN | 13.61% | 19.73% | **CORRECT** |
| 3 | FS | KIN | NON-KIN | 4.95% | 0.98% | **WRONG** |
| 4 | FD | NON-KIN | NON-KIN | 36.09% | 35.55% | **CORRECT** |
| 5 | FS | NON-KIN | KIN | 60.64% | 66.99% | **WRONG** |
| 6 | FS | KIN | KIN | 63.28% | 62.30% | **CORRECT** |
| 7 | MS | NON-KIN | KIN | 74.46% | 77.73% | **WRONG** |
| 8 | FS | KIN | KIN | 67.72% | 69.14% | **CORRECT** |
| 9 | FD | KIN | NON-KIN | 39.01% | 42.97% | **WRONG** |
| 10 | FD | KIN | NON-KIN | 34.59% | 35.74% | **WRONG** |

---

## Visualizations
The following plots are saved in this folder:
- **`visual_predictions.png`**: Side-by-side comparison of facial pairs showing actual vs predicted labels and correctness borders.
- **`confusion_matrix.png`**: Breakdown of model predictions vs actual categories.
- **`roc_curve.png`**: ROC Curve representing true positive vs false positive rate trade-offs.
