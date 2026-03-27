import numpy as np
from src.evaluation.metrics import print_metrics_binary


def test_perfect_predictions():
    y_true = [0, 0, 1, 1]
    predictions = np.array([0.0, 0.0, 1.0, 1.0])
    result = print_metrics_binary(y_true, predictions, verbose=0)
    assert result['auroc'] == 1.0
    assert result['acc'] == 1.0


def test_output_keys():
    y_true = [0, 1, 0, 1]
    predictions = np.array([0.3, 0.7, 0.4, 0.6])
    result = print_metrics_binary(y_true, predictions, verbose=0)
    expected_keys = {'acc', 'prec0', 'prec1', 'rec0', 'rec1', 'auroc', 'auprc', 'minpse'}
    assert set(result.keys()) == expected_keys
