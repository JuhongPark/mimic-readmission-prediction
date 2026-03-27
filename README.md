# MIMIC-III ICU Readmission Prediction

Predicting 30-day ICU readmission from MIMIC-III clinical records using an LSTM-CNN hybrid model. Combines clinical timeseries, disease embeddings, patient demographics, and optionally discharge note word vectors.

## Model

```
                        ┌─ Clinical timeseries  17ch × 48h, discretized at 1h
                        ├─ Disease embeddings   300-dim, ICD-9 claim codes
Input (390 / 590-dim) ─┤
                        ├─ Demographics         14-dim (age, gender, ethnicity, insurance)
                        └─ Word vectors         200-dim, discharge notes (optional)
                                  │
                            LSTM-CNN hybrid
                            (depth=2, dim=16)
                                  │
                          Binary prediction
                        (readmit within 30 days)
```

**Evaluation**: Accuracy, Precision, Recall, AUROC, AUPRC, min(Precision, Recall)

## Project Structure

```
config/       Hyperparameters and environment-configurable paths
src/          Core library — discretizer, data loading, evaluation, visualization
scripts/      Preprocessing and training entry points
notebooks/    SHAP explainability analysis
nlp/          BioWordVec embedding loader
data/         Data directory (see data/README.md for expected layout)
```

## Setup

**Prerequisites**

- Python 3.8+
- [MIMIC-III database access](https://physionet.org/content/mimiciii/) via PhysioNet
- [MIMIC-III ICU Readmission Analysis](https://github.com/JuhongPark/MIMIC-III_ICU_Readmission_Analysis) package
- BioWordVec embeddings (optional, for `train_generator_wordvec.py`)

```bash
pip install -e /path/to/MIMIC-III_ICU_Readmission_Analysis
pip install -r requirements.txt
export MIMIC_DATA_ROOT=/path/to/your/data
```

## Usage

### 1. Preprocess

```bash
python scripts/preprocess.py
```

### 2. Train

```bash
# Generator-based (memory efficient)
python scripts/train_generator.py

# In-memory (faster, needs more RAM)
python scripts/train_efficient.py

# With discharge note word vectors (input_dim=590)
python scripts/train_generator_wordvec.py
```

### 3. Interpret

SHAP analysis notebooks for model explainability:

- `notebooks/shap_pca_lstm.ipynb`
- `notebooks/shap_lstm_cnn.ipynb`

## License

MIT License - Copyright (c) 2020 Juhong Park
