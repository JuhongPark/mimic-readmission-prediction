import os

DATA_ROOT = os.environ.get("MIMIC_DATA_ROOT", "./data")

DATASET_DIR = os.path.join(DATA_ROOT, "created_readmission_data")
PREPROCESSED_DIR = os.path.join(DATA_ROOT, "preprocessed")
TRAIN_LIST = os.path.join(DATA_ROOT, "train_val_test", "train_listfile801010.csv")
TEST_LIST = os.path.join(DATA_ROOT, "train_val_test", "test_listfile801010.csv")
VAL_LIST = os.path.join(DATA_ROOT, "train_val_test", "val_listfile801010.csv")
OUTPUT_DIR = os.path.join(DATA_ROOT, "results")
NORMALIZER_PICKLE = os.path.join(DATA_ROOT, "normalizer.pickle")
DISCHARGE_WV_PICKLE = os.path.join(DATA_ROOT, "keyword_discharge_note_wv_v3.pk")
