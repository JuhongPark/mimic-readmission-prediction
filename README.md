# MIMIC-III ICU Readmission Prediction

> Predicting 30-day ICU readmission by fusing **four clinical data modalities** through an LSTM-CNN hybrid model, trained on [MIMIC-III](https://physionet.org/content/mimiciii/).

---

## Why

- Unplanned ICU readmissions → higher mortality, longer stays, increased costs
- Most models use structured data **or** clinical notes — rarely both
- This project combines **timeseries + diagnoses + demographics + discharge notes** into a single prediction

---

## Input Features

| Modality | What | Dim |
|:---------|:-----|----:|
| **Clinical timeseries** | 17 vital signs & lab values, discretized at 1h over 48h | 76 |
| **Disease embeddings** | ICD-9 codes → pre-trained medical concept vectors | 300 |
| **Demographics** | Age, gender, ethnicity, insurance (one-hot) | 14 |
| **Discharge notes** | [BioWordVec](https://github.com/ncbi-nlp/BioWordVec) embeddings *(optional)* | 200 |

**Key decisions:**
- Custom discretizer with 4 imputation strategies for irregular clinical sampling
- Balanced sampling to handle class imbalance (readmissions are rare)
- SHAP analysis for per-patient feature explanations

---

## Architecture

```
  Timeseries ──┐
  Diagnoses  ──┤  Concat     LSTM-CNN        Sigmoid
  Demographics ┤  per     →  (depth=2,  →    P(readmit
  Discharge  ──┘  timestep    dim=16)         ≤ 30 days)
  notes
```

**Metrics:** AUROC | AUPRC | Accuracy | Precision | Recall

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
```

---

## Quick Start

**Prerequisites:** Python 3.8+ · [MIMIC-III access](https://physionet.org/content/mimiciii/) · [benchmark package](https://github.com/JuhongPark/MIMIC-III_ICU_Readmission_Analysis)

```bash
pip install -e /path/to/MIMIC-III_ICU_Readmission_Analysis
pip install -r requirements.txt
export MIMIC_DATA_ROOT=/path/to/your/data
```

```bash
python scripts/preprocess.py                  # 1. Preprocess
python scripts/train_generator.py             # 2. Train (generator, memory efficient)
python scripts/train_generator_wordvec.py     # 2. Train (+ discharge note vectors)
```

See [`data/README.md`](data/README.md) for expected data layout.

---

MIT License - Copyright (c) 2020 Juhong Park
