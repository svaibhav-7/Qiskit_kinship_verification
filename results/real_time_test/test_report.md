# KinFaceW-I Real-Time Evaluation Report

**Generated Date:** 2026-05-30 13:35:23
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
| 1 | FS | KIN | KIN | 71.98% | 67.77% | **CORRECT** |
| 2 | FD | KIN | NON-KIN | 39.01% | 37.50% | **WRONG** |
| 3 | MS | NON-KIN | NON-KIN | 21.57% | 16.99% | **CORRECT** |
| 4 | FS | NON-KIN | NON-KIN | 13.87% | 11.52% | **CORRECT** |
| 5 | FS | KIN | NON-KIN | 4.95% | 9.77% | **WRONG** |
| 6 | FS | KIN | KIN | 63.28% | 58.40% | **CORRECT** |
| 7 | FD | KIN | KIN | 58.00% | 57.62% | **CORRECT** |
| 8 | FD | NON-KIN | KIN | 51.15% | 49.02% | **WRONG** |
| 9 | FD | KIN | KIN | 90.73% | 89.65% | **CORRECT** |
| 10 | FS | NON-KIN | KIN | 59.77% | 57.23% | **WRONG** |
| 11 | FD | NON-KIN | KIN | 54.29% | 51.76% | **WRONG** |
| 12 | FS | KIN | KIN | 67.72% | 70.31% | **CORRECT** |
| 13 | MS | NON-KIN | KIN | 74.46% | 75.59% | **WRONG** |
| 14 | FD | KIN | NON-KIN | 34.59% | 30.66% | **WRONG** |
| 15 | FD | NON-KIN | NON-KIN | 36.09% | 31.84% | **CORRECT** |

---

## Visualizations
The following plots are saved in this folder:
- **`visual_predictions.png`**: Side-by-side comparison of facial pairs showing actual vs predicted labels and correctness borders.
- **`confusion_matrix.png`**: Breakdown of model predictions vs actual categories.
- **`roc_curve.png`**: ROC Curve representing true positive vs false positive rate trade-offs.
