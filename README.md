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

**ML design tradeoffs**

| Decision | Tradeoff |
|:---------|:---------|
| **Broadcast static features** | Diagnoses and demographics are copied to every timestep as constant context. Simple, but assumes no temporal interaction with these signals. |
| **Imputation as prior** | Forward fill assumes "last value persists." The model learns from this assumption, not raw observations — imputation strategy shapes what the model sees. |
| **Balanced sampling** | Equalizes classes each epoch → better decision boundary, but outputs are no longer calibrated to true prevalence. Deployment requires recalibration. |

**ML achievements & implications**

| Area | Detail |
|:-----|:-------|
| **Multimodal clinical fusion** | Fuses four heterogeneous sources (timeseries, ICD-9 embeddings, demographics, discharge notes) into a 390-dimensional temporal input (590 with discharge notes). Each modality captures a different clinical signal — vitals trajectory, diagnostic history, patient context, and narrative clinical reasoning — that a single source cannot provide alone. |
| **Biomedical NLP pipeline** | Extracts drug and disease entities from 12 discharge note sections via BC5CDR NER, then maps them to 200-dimensional BioWordVec embeddings (pre-trained on PubMed + MIMIC-III). Converts unstructured clinical text into dense vector features without task-specific fine-tuning. |
| **Explainability for clinical adoption** | Physicians initially declined to use the model because they could not trust a black-box prediction. Integrating SHAP (DeepExplainer, KernelExplainer) to produce per-feature attribution with force plots and summary plots changed that — once clinicians could inspect which variables drove each prediction and verify alignment with medical knowledge, they were willing to incorporate the model into their decision-making. Interpretability was not a nice-to-have; it was the prerequisite for clinical adoption. |
| **End-to-end reproducibility** | Environment-configurable paths, Keras Sequence generators for memory-efficient training, and a modular pipeline (preprocess → train → evaluate → explain) enable full reproduction from raw MIMIC-III data to SHAP visualizations. |

**Interpretability**

| Issue | Why it matters |
|:------|:---------------|
| **Temporal attribution ambiguity** | Static features broadcast to all timesteps receive timestep-specific SHAP values — but that temporal variation comes from the model's hidden state, not from the input changing. "Which modality" and "which timestep" are conflated. |
| **Imputed value attribution** | SHAP attributes importance to forward-filled values that were never measured. A mask channel flags observation status, but feature-level attribution still operates on the imputed input — what the model explains ≠ what was clinically observed. |

**AI safety & security considerations**

| Concern | Detail |
|:--------|:-------|
| **De-identified data & access control** | MIMIC-III requires PhysioNet credentialed access and CITI ethics training. Records are de-identified, but derived artifacts (embeddings, SHAP values) may carry re-identification risk when linked with auxiliary data — model outputs warrant the same access controls as the source data. |
| **Explainability as a safety mechanism** | SHAP attribution provides an audit trail per prediction. Physician collaboration showed that without this layer, clinicians reject the model entirely — explainability is not just a technical feature but the trust boundary that determines whether an AI system is used or ignored in practice. |
| **Calibration & decision safety** | Balanced sampling shifts the decision threshold away from true prevalence. Without recalibration, deployment risks over-alerting (wasted clinical resources) or under-alerting (missed readmissions). Threshold tuning on prevalence-representative data is required. |
| **Metric ≠ objective** | High AUROC on balanced data does not guarantee that acting on predictions reduces readmissions. The metric optimized and the clinical outcome desired are not the same. |
| **Encoded disparities** | Demographic features reflect historical care patterns. Uneven subgroup performance may stem from the data, not the model — fairness auditing across subgroups is required before deployment. |
| **NLP pipeline integrity** | Discharge note parsing relies on section header matching — non-standard formatting yields empty or misaligned entity extractions. Silent failures propagate incorrect feature vectors without error signals, requiring input validation at the pipeline boundary. |
