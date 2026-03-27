"""
Train LSTM-CNN model for ICU readmission prediction.
Generator version: uses Keras Sequence for memory-efficient data loading.
"""
import argparse
import os
import numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras.optimizers import Adam
from mimic3models.common_keras_models import lstm_cnn
from mimic3models.preprocessing import Normalizer
from src.evaluation.metrics import print_metrics_binary, save_results
from utilities.data_loader import get_embeddings

from config.paths import (
    TRAIN_LIST, TEST_LIST, VAL_LIST, OUTPUT_DIR, NORMALIZER_PICKLE,
)
from config.defaults import AGE_MEANS, AGE_STD, CONT_CHANNELS, DEFAULT_HPARAMS
from src.discretizer import StandardDiscretizer
from src.data.generator import Generator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LSTM-CNN for ICU readmission prediction",
    )
    parser.add_argument("--epochs", type=int, default=None, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001)")
    parser.add_argument("--dim", type=int, default=None, help="LSTM hidden dimension")
    parser.add_argument("--dropout", type=float, default=None, help="dropout rate")
    parser.add_argument("--output-dir", type=str, default=None, help="output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    hp = DEFAULT_HPARAMS.copy()
    for key in ['epochs', 'batch_size', 'dim', 'dropout']:
        val = getattr(args, key, None)
        if val is not None:
            hp[key] = val
    output_path = args.output_dir or OUTPUT_DIR

    # Load embeddings and normalizer
    normalizer = Normalizer(fields=CONT_CHANNELS)
    normalizer.load_params(NORMALIZER_PICKLE)
    embeddings, word_indices = get_embeddings(corpus='claims_codes_hs', dim=300)
    discretizer = StandardDiscretizer(
        timestep=float(hp['timestep']), store_masks=True,
        impute_strategy='previous', start_time='zero',
    )

    # Create generators
    train_generator = Generator(
        mode='train', data_dir=TRAIN_LIST, batch_size=hp['batch_size'],
        normalizer=normalizer, discretizer=discretizer,
        embeddings=embeddings, word_indices=word_indices,
    )
    val_generator = Generator(
        mode='val', data_dir=VAL_LIST, batch_size=1,
        normalizer=normalizer, discretizer=discretizer,
        embeddings=embeddings, word_indices=word_indices,
    )
    print('train data size:', train_generator.size())
    print('validation data size:', val_generator.size())

    # Build model
    print("==> using model {}".format(hp['network']))
    model = lstm_cnn.Network(
        dim=hp['dim'], batch_norm=True, dropout=hp['dropout'],
        rec_dropout=hp['rec_dropout'], task=hp['task'],
        target_repl=hp['target_repl'], deep_supervision=False,
        num_classes=1, depth=hp['depth'], input_dim=390,
    )
    model.final_name = model.say_name()
    print("==> model.final_name:", model.final_name)
    model.compile(
        optimizer=Adam(learning_rate=args.lr, beta_1=0.9),
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

    # Test
    model.load_weights(path + '.hdf5')
    test_generator = Generator(
        mode='test', data_dir=TEST_LIST, batch_size=1,
        normalizer=normalizer, discretizer=discretizer,
        embeddings=embeddings, word_indices=word_indices,
    )
    predictions = model.predict(test_generator, steps=test_generator.size())
    test_labels = test_generator.data_list['y_true']
    test_names = test_generator.whole_names

    predictions = np.array(predictions)[:, 0]
    results = print_metrics_binary(test_labels, predictions)

    result_path = os.path.join(output_path, "test_predictions.csv")
    save_results(results, test_names, predictions, test_labels, result_path)


if __name__ == '__main__':
    main()
