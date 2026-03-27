from __future__ import annotations
import numpy as np
import pandas as pd


def disease_embedding(
    embeddings: np.ndarray, word_indices: dict[str, int], diseases_list: list[list[str]]
) -> list[list[float]]:
    emb_list = []
    for diseases in diseases_list:
        emb_period = [0] * 300
        skip = 0
        for disease in diseases:
            k = 'IDX_' + str(disease)
            if k not in word_indices.keys():
                skip += 1
                continue
            index = word_indices[k]
            emb_disease = embeddings[index]
            emb_period = [sum(x) for x in zip(emb_period, emb_disease)]
        found = len(diseases) - skip
        emb_period = [x / found for x in emb_period] if found > 0 else emb_period
        emb_list.append(emb_period)
    return emb_list


def get_wordvectors(names: list[str], discharge_wv: pd.DataFrame) -> list:
    wordvector_list = []
    namelist = []
    for element in names:
        x = element.split('_')
        namelist.append(x[2])
    for icustay in namelist:
        matched = discharge_wv[discharge_wv.ICUSTAY_ID == icustay].k500_mean_wv
        if len(matched) == 0:
            wordvector_list.append([0] * 200)
        else:
            wordvector_list.append(matched.iloc[0].reshape((-1, 1)))
    return wordvector_list


def age_normalize(demographic: list[list[float]], age_means: float, age_std: float) -> list:
    demographic = np.asmatrix(demographic)
    demographic[:, 0] = (demographic[:, 0] - age_means) / age_std
    return demographic.tolist()
