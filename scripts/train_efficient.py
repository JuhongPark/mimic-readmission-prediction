"""
Train LSTM-CNN model for ICU readmission prediction.
Efficient version: loads all data into memory at once.
"""
import os
import numpy as np
import pandas as pd

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from mimic3models.common_keras_models import lstm_cnn
from mimic3models.preprocessing import Normalizer
from mimic3models import keras_utils
from src.evaluation.metrics import print_metrics_binary, save_results
from utilities.data_loader import get_embeddings

from config.paths import (
    DATASET_DIR, TRAIN_LIST, TEST_LIST, VAL_LIST, OUTPUT_DIR, NORMALIZER_PICKLE,
)
from config.defaults import AGE_MEANS, AGE_STD, CONT_CHANNELS, DEFAULT_HPARAMS
from src.discretizer import StandardDiscretizer
from src.data.reader import read_example, get_diseases, get_demographic
from src.data.features import disease_embedding, age_normalize
from src.data.padding import pad_zeros


def load_data(list_dir, discretizer, normalizer, embeddings, word_indices):
    """Load and preprocess a full dataset into memory."""
    data_list = pd.read_csv(list_dir)
    batch_dict = {}
    for i in range(len(data_list)):
        row = data_list.loc[i]
        ret = read_example(row['stay'], row['period_length'], row['y_true'])
        for k, v in ret.items():
            if k not in batch_dict:
                batch_dict[k] = []
            batch_dict[k].append(v)
    batch_dict["header"] = batch_dict["header"][0]

    data = batch_dict["X"]
    ts = batch_dict["t"]
    data = [discretizer.transform_end_t_hours(X, los=t)[0] for (X, t) in zip(data, ts)]

    labels = batch_dict["y"]
    names = batch_dict["name"]
    diseases_list = get_diseases(names)
    diseases_emb = disease_embedding(embeddings, word_indices, diseases_list)
    demographic = get_demographic(names)
    demographic = age_normalize(demographic, AGE_MEANS, AGE_STD)

    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    data = [np.hstack([X, [d] * len(X)]) for (X, d) in zip(data, diseases_emb)]
    data = [np.hstack([X, [d] * len(X)]) for (X, d) in zip(data, demographic)]
    pad_data = pad_zeros(data)

    return pad_data, np.array(labels), names


def main():
    hp = DEFAULT_HPARAMS.copy()
    hp['batch_size'] = 8
    output_path = OUTPUT_DIR

    # Load embeddings and normalizer
    normalizer = Normalizer(fields=CONT_CHANNELS)
    normalizer.load_params(NORMALIZER_PICKLE)
    embeddings, word_indices = get_embeddings(corpus='claims_codes_hs', dim=300)
    discretizer = StandardDiscretizer(
        timestep=float(hp['timestep']), store_masks=True,
        impute_strategy='previous', start_time='zero',
    )

    # Load datasets
    print("==> Loading training data...")
    train_data, train_labels, train_names = load_data(
        TRAIN_LIST, discretizer, normalizer, embeddings, word_indices,
    )
    train_raw = (train_data, train_labels)

    print("==> Loading validation data...")
    val_data, val_labels, val_names = load_data(
        VAL_LIST, discretizer, normalizer, embeddings, word_indices,
    )
    val_raw = (val_data, val_labels)

    # Build model
    print("==> using model {}".format(hp['network']))
    model = lstm_cnn.Network(
        dim=hp['dim'], batch_norm=True, dropout=hp['dropout'],
        rec_dropout=hp['rec_dropout'], task=hp['task'],
        target_repl=hp['target_repl'], deep_supervision=False,
        num_classes=1, depth=hp['depth'], input_dim=390,
    )
    suffix = ".bs{}{}{}.ts{}{}".format(
        2,
        ".L1{}".format(hp['l1']) if hp['l1'] > 0 else "",
        ".L2{}".format(hp['l2']) if hp['l2'] > 0 else "",
        hp['timestep'],
        ".trc{}".format(hp['target_repl_coef']) if hp['target_repl_coef'] > 0 else "",
    )
    model.final_name = model.say_name() + suffix
    print("==> model.final_name:", model.final_name)
    model.compile(optimizer=Adam(learning_rate=0.001, beta_1=0.9), loss=hp['loss'], loss_weights=hp['loss_weights'])
    model.summary()

    # Prepare callbacks
    path = os.path.join(output_path, 'keras_state', model.final_name + '.epoch{epoch}.test{val_loss}.state')
    os.makedirs(os.path.dirname(path), exist_ok=True)

    metrics_callback = keras_utils.ReadmissionMetrics(
        train_data=train_raw, val_data=val_raw,
        target_repl=(hp['target_repl_coef'] > 0), batch_size=2, verbose=2,
    )
    saver = ModelCheckpoint(path, verbose=1, period=20)

    os.makedirs('keras_logs', exist_ok=True)
    csv_logger = CSVLogger(
        os.path.join('keras_logs', model.final_name + '.csv'),
        append=True, separator=';',
    )

    # Train
    print("==> training")
    model.fit(
        x=train_raw[0], y=train_raw[1],
        validation_data=val_raw,
        nb_epoch=hp['epochs'],
        callbacks=[metrics_callback, saver, csv_logger],
        shuffle=True, verbose=2, batch_size=2,
    )

    # Test
    print("==> Loading test data...")
    test_data, test_labels, test_names = load_data(
        TEST_LIST, discretizer, normalizer, embeddings, word_indices,
    )
    predictions = model.predict(test_data, batch_size=1, verbose=1)
    predictions = np.array(predictions)[:, 0]
    results = print_metrics_binary(test_labels, predictions)

    result_path = os.path.join(output_path, "test_predictions.csv")
    save_results(results, test_names, predictions, test_labels, result_path)


if __name__ == '__main__':
    main()
