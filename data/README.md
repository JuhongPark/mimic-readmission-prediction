## Data Directory

Place your MIMIC-III derived data here. Expected structure:

```
data/
├── created_readmission_data/     # Discretized timeseries CSV files
├── preprocessed/                 # Per-subject directories with diagnoses, demographics
├── train_val_test/
│   ├── train_listfile801010.csv  # Training split
│   ├── val_listfile801010.csv    # Validation split
│   └── test_listfile801010.csv   # Test split
├── normalizer.pickle             # Pre-computed normalization parameters
└── keyword_discharge_note_wv_v3.pk  # Discharge note word vectors (optional)
```

Set `MIMIC_DATA_ROOT` environment variable to override the default data path.
