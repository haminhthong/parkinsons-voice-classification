# 📋 Model Card: Parkinson’s Voice Feature Screening Prototype

## 1. Model Overview & Positioning

- **Model Name:** Parkinson’s Voice Feature Screening Pipeline (Leakage-Aware Prototype)
- **Version:** `1.2.0`
- **Model Architecture:** `StandardScaler` $\to$ `SelectKBest(k=10, f_classif)` $\to$ `KNeighborsClassifier(n_neighbors=9, p=1, weights='distance')` with Group-Aware Sigmoid Calibration (`CalibratedClassifierCV`) and Subject-Level `median` Aggregation.
- **Input Scope:** 22 pre-extracted tabular acoustic voice features (fundamental frequency, jitter, shimmer, harmonic-to-noise ratios, nonlinear dynamical measures) from sustained phonation `/a/` (UCI Parkinsons).
- **Audio Limitation:** **The model does NOT process raw audio files (WAV, MP3, FLAC).** It operates strictly as a tabular acoustic feature screening model.

---

## 2. Intended Use & Clinical Boundaries

### ✅ Intended Uses:
- **Academic & Portfolio Research:** Demonstrating leakage-aware validation architectures, patient-level stratification, out-of-fold calibration, and cluster bootstrap uncertainty on grouped biomedical tabular data.
- **Experimental Screening Signal:** Generating continuous risk scores (`screening_score`) and categorical screening flags (`model-positive`, `model-negative`) under explicit research caveats.

### ❌ Non-Intended Uses:
- **NOT a Medical Diagnostic System:** The model cannot diagnose Parkinson’s Disease or replace neurological examination, dopamine transporter imaging (DaTscan), or clinical motor scoring (MDS-UPDRS).
- **NOT an End-to-End Voice Diagnosis Tool:** It does not extract features from raw microphone recordings or perform acoustic signal processing.
- **NOT Validated for Clinical Deployment:** The prototype has not undergone clinical trial validation, multi-site external validation, or regulatory clearance (e.g., FDA 510(k), CE-MDR).

---

## 3. Training & Evaluation Data

- **Dataset Source:** [UCI Machine Learning Repository: Parkinsons Telemonitoring / Voice Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons).
- **Cohort Size:** 195 acoustic recordings from **32 distinct subjects** (24 diagnosed with Parkinson's Disease, 8 Healthy Controls). Each subject provided 5–6 sustained vowel phonations.
- **Data Partitioning (Zero-Leakage Invariant):**
  - **Training Cohort:** 24 subjects (18 PD, 6 Control) $\to$ 147 recordings.
  - **Holdout Test Cohort:** 8 subjects (6 PD, 2 Control) $\to$ 48 recordings.
  - **Constraint:** Complete subject isolation ($\text{Train Subjects} \cap \text{Holdout Subjects} = \emptyset$). No recordings from the same individual cross split boundaries.

---

## 4. Canonical 8-Stage Architecture

```
1. DATA INGESTION
   UCI Parkinsons Tabular Acoustic Features (195 recordings / 32 subjects)
          ↓
2. SUBJECT IDENTITY & SCHEMA AUDIT
   Schema validation, subject_id extraction, remove algebraic redundancies (Jitter:DDP, Shimmer:DDA)
          ↓
3. PATIENT-LEVEL HOLDOUT SPLIT
   24 Train Subjects / 8 Independent Holdout Subjects (Stratified, Zero Leakage)
          ↓
4. MODEL DEVELOPMENT INSIDE TRAIN
   Subject-Stratified Folds → Fold-Safe Pipeline (Scaler → SelectKBest → Classifier)
          ↓
5. ROBUST MODEL SELECTION
   Nested Subject-Level CV with Stability & Simplicity Guardrails (tie-breaker for equal F1)
          ↓
6. PROBABILITY & DECISION LAYER
   Group-Aware Sigmoid Calibration → OOF Probabilities → Aggregation (median) → OOF Threshold Optimization
          ↓
7. FINAL HOLDOUT EVALUATION
   Single Evaluation on 8 Unseen Subjects → Metrics + Patient-Cluster Bootstrap 95% CI (2,000x)
          ↓
8. SERVING & RELIABILITY LAYER
   FastAPI & Streamlit → P1-P99 OOD Range Checks → Minimum Recordings Policy (<3 recordings flagged)
```

---

## 5. Performance Metrics & Statistical Uncertainty

### ⚠️ Critical Sample Size & Holdout Caveats
The independent holdout cohort contains only **8 patients (6 PD, 2 Healthy Controls)**.
- **Specificity = 0.50** corresponds to correctly identifying only **1 out of 2 control individuals**.
- **ROC-AUC = 1.00** on 8 subjects is a descriptive sample statistic, not conclusive proof of flawless discrimination.
- **Expected Calibration Error (ECE = 0.0513 with 5 bins)** on 8 subjects is purely descriptive due to sparse sample counts per bin.
- **Nested Cross-Validation** on 24 subjects reveals substantial uncertainty ($F_1\text{-macro} = 0.519 \pm 0.328$, Balanced Accuracy = 0.608) and provides a much more honest and conservative indication of expected real-world generalization than the holdout point estimate alone.

### Performance Summary Table

| Evaluation Layer | Cohort | F1-Macro | Balanced Accuracy | Sensitivity | Specificity | ROC-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Inner Model Search** | 24 Train Subjects (5-Fold CV) | 0.7270 | 0.7500 | 0.8889 | 0.6000 | 0.9000 | — |
| **Nested Subject CV** | 24 Train Subjects (5 Outer $\times$ 3 Inner) | 0.5190 ± 0.328 | 0.6083 | 0.6444 | 0.5722 | 0.6333 | 0.1893 |
| **Final Holdout Point Estimate** | 8 Unseen Subjects | 0.7949 | 0.7500 | 1.0000 | 0.5000 | 1.0000 | 0.1124 |
| **Patient-Cluster Bootstrap 95% CI** | 8 Unseen Subjects (2,000 replicates) | **[0.385, 1.000]** | **[0.500, 1.000]** | **[1.000, 1.000]** | **[0.000, 1.000]** | **[1.000, 1.000]** | **[0.034, 0.269]** |

---

## 6. Serving Guardrails & Out-of-Distribution Policy

To prevent silent failures and deceptive predictions during inference, the serving runtime incorporates three automated guardrails:

1. **P1–P99 Feature Distribution Checks:**
   During training, the 1st and 99th percentiles ($P_1, P_{99}$) of all 20 modeling features are recorded. If any input feature falls outside $[P_1, P_{99}]$, the system emits a `FEATURE_OUTSIDE_TRAINING_RANGE` warning and marks subject reliability as `"limited"`.
2. **Minimum Recordings Policy:**
   Because training subjects contributed 5–6 recordings each, computing subject aggregations on $<3$ recordings carries elevated variance. When fewer than 3 recordings are supplied, the system issues an `ONLY_ONE_RECORDING` warning and classifies reliability as `"limited"`.
3. **Training Label Rejection:**
   The inference API schema (`POST /predict/subject`) specifies `extra = "forbid"` and actively rejects payloads containing the training target `status` with a `422 Unprocessable Entity` status.

---

## 7. Limitations & Technical Debt

1. **Severe Sample Size Limitations:** 32 subjects total (only 8 healthy controls in the entire dataset). Small validation folds are prone to high metric variance.
2. **Lack of Demographic Covariates:** The UCI dataset omits age, biological sex, recording hardware, and clinical site metadata. Subgroup fairness and demographic parity cannot be audited.
3. **Absence of External Cohort Validation:** The pipeline has not been tested against external speech datasets (e.g., PC-GITA, mPower) due to acoustic schema differences.
4. **Artifact Serialization Security:** Current deployment bundles use `joblib`. While standard for local portfolios, enterprise deployments should migrate to hardened formats like `skops` or `ONNX` to eliminate arbitrary code execution vulnerabilities.
