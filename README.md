# MIMIC-III ICU Readmission Prediction

> Predicting 30-day ICU readmission by fusing **four clinical data modalities** through an LSTM-CNN hybrid model, trained on [MIMIC-III](https://physionet.org/content/mimiciii/).

---

## Why

- Unplanned ICU readmissions → higher mortality, longer stays, increased costs
- Single-modality models miss complementary signals across vitals, diagnoses, and clinical text
- This project fuses **four heterogeneous data sources** into a unified temporal representation for prediction

---

## Input Features

| Modality | What | Dim |
|:---------|:-----|----:|
| **Clinical timeseries** | 17 vital signs & lab values, discretized at 1h over 48h | 76 |
| **Disease embeddings** | ICD-9 codes → pre-trained medical concept vectors | 300 |
| **Demographics** | Age, gender, ethnicity, insurance (one-hot) | 14 |
| **Discharge notes** | [BioWordVec](https://github.com/ncbi-nlp/BioWordVec) embeddings *(optional)* | 200 |

**Key decisions:**
- Custom discretizer with 4 imputation strategies — clinical timeseries are irregularly sampled and missing-data handling directly impacts model quality
- Balanced sampling — readmissions are rare events, naive training biases toward the majority class
- SHAP explainability — interpretability matters for clinical decision support

---

## Architecture

```
[Timeseries   76d] ─┐
[Diagnoses   300d] ──┼── Concat per timestep ── LSTM-CNN (depth=2, dim=16) ── P(readmit ≤ 30d)
[Demographics 14d] ──┤
[Discharge   200d] ──┘   (390-dim, or 590 with discharge notes)
```

Time-invariant features (diagnoses, demographics) are broadcast across all 48 timesteps before concatenation.

**Evaluation:** AUROC | AUPRC | Accuracy | Precision | Recall

---

## Project Structure

```
config/          Hyperparameters, environment-configurable paths
src/
├── discretizer/   Timeseries discretization (17ch / 50ch)
├── data/          Loading, feature engineering, Keras Sequence generator
├── evaluation/    Classification metrics, result export
└── visualization/ Training curves, ROC plots
scripts/         Preprocess + 3 training variants
notebooks/       SHAP explainability, NER analysis
nlp/             BioWordVec loader
tests/           Unit tests for core utilities
```

---

## Quick Start

**Prerequisites:** Python 3.8+ · [MIMIC-III access](https://physionet.org/content/mimiciii/) · [benchmark package](https://github.com/JuhongPark/MIMIC-III_ICU_Readmission_Analysis)

```bash
pip install -e /path/to/MIMIC-III_ICU_Readmission_Analysis
pip install -e ".[dev]"
export MIMIC_DATA_ROOT=/path/to/your/data

python scripts/preprocess.py                              # 1. Preprocess
python scripts/train_generator.py                         # 2. Train (defaults)
python scripts/train_generator.py --epochs 100 --lr 5e-4  #    (custom)
python scripts/train_generator_wordvec.py                 #    (+ discharge note vectors)
pytest                                                    # 3. Run tests
```

See [`data/README.md`](data/README.md) for expected data layout.

---

## Discussion

**ML**
- Multi-modal alignment is a design choice, not just concatenation. Broadcasting time-invariant features (diagnoses, demographics) across timesteps treats them as constant context — a simplification that works here but assumes these signals don't interact with temporal dynamics.
- The discretizer encodes clinical domain knowledge directly into preprocessing. Imputation strategy (forward fill vs. zero) is effectively a prior about how missing clinical data should behave — and the model learns from that prior, not from the raw observations.
- Balanced sampling trades calibration for discrimination. The generator resamples each epoch to equalize classes, which improves the decision boundary but means model outputs diverge from true readmission probability. Any deployment would require post-hoc recalibration.

**Interpretability**
- SHAP on multi-modal temporal inputs raises an attribution ambiguity: when static features (disease embeddings) are broadcast to every timestep, SHAP may assign them timestep-specific importance that has no meaningful temporal interpretation. Disentangling "which modality mattered" from "which timestep mattered" requires modality-level ablation beyond standard feature attribution.
- Forward-fill imputation creates a subtler problem — SHAP can attribute importance to imputed values that were never actually measured. Attribution points to what the model sees, not what was clinically observed. This gap between model-level and ground-truth explanation is worth surfacing to end users.

**Deployment**
- Optimizing AUROC on balanced data may not align with the actual clinical goal. High discriminative performance doesn't guarantee that acting on the model's predictions reduces readmissions — the metric optimized and the outcome desired are not the same objective.
- Demographic features (ethnicity, insurance) reflect systemic patterns in the training data. The model may perform unevenly across subgroups not because of a modeling error, but because the data itself encodes unequal care history.

---

MIT License - Copyright (c) 2020 Juhong Park
