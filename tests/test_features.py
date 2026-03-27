import numpy as np
from src.data.features import disease_embedding, age_normalize


def test_disease_embedding_averages_found_codes():
    embeddings = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    word_indices = {'IDX_001': 0, 'IDX_002': 1}
    result = disease_embedding(embeddings, word_indices, [['001', '002']])
    expected = [2.5, 3.5, 4.5]
    assert len(result) == 1
    assert np.allclose(result[0][:3], expected)


def test_disease_embedding_skips_unknown_codes():
    embeddings = np.array([[1.0, 2.0, 3.0]])
    word_indices = {'IDX_001': 0}
    result = disease_embedding(embeddings, word_indices, [['001', '999']])
    assert np.allclose(result[0][:3], [1.0, 2.0, 3.0])


def test_disease_embedding_all_unknown():
    embeddings = np.array([[1.0, 2.0, 3.0]])
    word_indices = {'IDX_001': 0}
    result = disease_embedding(embeddings, word_indices, [['999']])
    assert all(v == 0 for v in result[0])


def test_age_normalize_only_first_column():
    demographic = [[80.0, 1, 0, 1], [60.0, 0, 1, 0]]
    result = age_normalize(demographic, age_means=70.0, age_std=10.0)
    assert np.isclose(result[0][0], 1.0)
    assert np.isclose(result[1][0], -1.0)
    assert result[0][1] == 1
    assert result[1][2] == 1
