# KinFaceW-I Real-Time Evaluation Report

**Generated Date:** 2026-05-30 14:00:45
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
| 1 | FS | KIN | KIN | 71.98% | 71.09% | **CORRECT** |
| 2 | FD | KIN | NON-KIN | 39.01% | 38.28% | **WRONG** |
| 3 | MS | NON-KIN | NON-KIN | 21.57% | 22.46% | **CORRECT** |
| 4 | FS | NON-KIN | NON-KIN | 13.87% | 7.62% | **CORRECT** |
| 5 | FS | KIN | NON-KIN | 4.95% | 0.39% | **WRONG** |
| 6 | FS | KIN | KIN | 63.28% | 60.16% | **CORRECT** |
| 7 | FD | KIN | KIN | 58.00% | 58.79% | **CORRECT** |
| 8 | FD | NON-KIN | KIN | 51.15% | 51.37% | **WRONG** |
| 9 | FD | KIN | KIN | 90.73% | 90.82% | **CORRECT** |
| 10 | FS | NON-KIN | KIN | 59.77% | 59.18% | **WRONG** |
| 11 | FD | NON-KIN | KIN | 54.29% | 55.08% | **WRONG** |
| 12 | FS | KIN | KIN | 67.72% | 69.53% | **CORRECT** |
| 13 | MS | NON-KIN | KIN | 74.46% | 72.46% | **WRONG** |
| 14 | FD | KIN | NON-KIN | 34.59% | 32.81% | **WRONG** |
| 15 | FD | NON-KIN | NON-KIN | 36.09% | 34.18% | **CORRECT** |

---

## Visualizations
The following plots are saved in this folder:
- **`visual_predictions.png`**: Side-by-side comparison of facial pairs showing actual vs predicted labels and correctness borders.
- **`confusion_matrix.png`**: Breakdown of model predictions vs actual categories.
- **`roc_curve.png`**: ROC Curve representing true positive vs false positive rate trade-offs.
