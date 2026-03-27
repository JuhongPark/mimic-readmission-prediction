"""
Train LSTM-CNN model for ICU readmission prediction.
Generator version with word vectors: adds 200-dim discharge note word vectors
as extra features (input_dim=590 instead of 390).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras.optimizers import Adam
from mimic3models.common_keras_models import lstm_cnn
from mimic3models.preprocessing import Normalizer
from utilities.data_loader import get_embeddings

from config.paths import (
    TRAIN_LIST, TEST_LIST, VAL_LIST, OUTPUT_DIR,
    NORMALIZER_PICKLE, DISCHARGE_WV_PICKLE,
)
from config.defaults import AGE_MEANS, AGE_STD, CONT_CHANNELS, DEFAULT_HPARAMS
from src.discretizer import StandardDiscretizer
from src.data.generator import Generator
from src.evaluation.metrics import print_metrics_binary, save_results
from src.visualization.plots import plot_training_history, plot_roc_curve


def main():
    hp = DEFAULT_HPARAMS.copy()
    hp['batch_size'] = 100
    experiment = 'experiment5'
    output_path = os.path.join(OUTPUT_DIR, experiment)

    # Load embeddings and normalizer
    normalizer = Normalizer(fields=CONT_CHANNELS)
    normalizer.load_params(NORMALIZER_PICKLE)
    embeddings, word_indices = get_embeddings(corpus='claims_codes_hs', dim=300)
    discretizer = StandardDiscretizer(
        timestep=float(hp['timestep']), store_masks=True,
        impute_strategy='previous', start_time='zero',
    )

    # Load discharge word vectors
    discharge_wv = pd.read_pickle(DISCHARGE_WV_PICKLE)

    # Create generators
    train_generator = Generator(
        mode='train', data_dir=TRAIN_LIST, batch_size=hp['batch_size'],
        normalizer=normalizer, discretizer=discretizer,
        embeddings=embeddings, word_indices=word_indices,
        discharge_wv=discharge_wv,
    )
    val_generator = Generator(
        mode='val', data_dir=VAL_LIST, batch_size=1,
        normalizer=normalizer, discretizer=discretizer,
        embeddings=embeddings, word_indices=word_indices,
        discharge_wv=discharge_wv,
    )
    print('train data size:', train_generator.size())
    print('validation data size:', val_generator.size())

    # Build model (input_dim=590 = 390 base + 200 word vectors)
    print("==> using model {}".format(hp['network']))
    model = lstm_cnn.Network(
        dim=hp['dim'], batch_norm=True, dropout=hp['dropout'],
        rec_dropout=hp['rec_dropout'], task=hp['task'],
        target_repl=hp['target_repl'], deep_supervision=False,
        num_classes=1, depth=hp['depth'], input_dim=590,
    )
    model.final_name = model.say_name()
    print("==> model.final_name:", model.final_name)
    model.compile(
        optimizer=Adam(learning_rate=0.001, beta_1=0.9),
        loss=hp['loss'], metrics=['acc'], loss_weights=hp['loss_weights'],
    )
    model.summary()

    # Prepare training
    path = os.path.join(output_path, 'keras_state', model.final_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=path + '.hdf5', save_best_only=True,
        monitor='val_acc', save_weights_only=True, mode='max', period=1,
    )

    hist = model.fit(
        train_generator, verbose=1, epochs=hp['epochs'],
        steps_per_epoch=train_generator.size() // hp['batch_size'],
        validation_steps=val_generator.size(),
        validation_data=val_generator,
        callbacks=[checkpoint],
    )

    h = hist.history
    plot_training_history(h, hp['epochs'], hp['batch_size'], output_path, model.final_name)

    # Test
    model.load_weights(path + '.hdf5')
    test_generator = Generator(
        mode='test', data_dir=TEST_LIST, batch_size=1,
        normalizer=normalizer, discretizer=discretizer,
        embeddings=embeddings, word_indices=word_indices,
        discharge_wv=discharge_wv,
    )
    predictions = model.predict(test_generator, steps=test_generator.size())
    predictions = np.array(predictions)[:, 0]
    test_labels = test_generator.data_list['y_true']
    test_names = test_generator.whole_names

    plot_roc_curve(test_labels, predictions, model.final_name, output_path)

    results = print_metrics_binary(test_labels, predictions)
    result_path = os.path.join(output_path, "test_predictions.csv")
    save_results(results, test_names, predictions, test_labels, result_path)


if __name__ == '__main__':
    main()
