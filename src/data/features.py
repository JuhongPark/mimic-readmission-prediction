import numpy as np


def disease_embedding(embeddings, word_indices, diseases_list):
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
        emb_period = [x / len(diseases) for x in emb_period]
        emb_list.append(emb_period)
    return emb_list


def get_wordvectors(names, discharge_wv):
    wordvector_list = []
    namelist = []
    for element in names:
        x = element.split('_')
        namelist.append(x[2])
    for icustay in namelist:
        wordvector = discharge_wv[discharge_wv.ICUSTAY_ID == icustay].k500_mean_wv
        if len(wordvector) != 200:
            wordvector = [0] * 200
            wordvector_list.append(wordvector)
        else:
            wordvector_list.append(wordvector[0].reshape((-1, 1)))
    return wordvector_list


def age_normalize(demographic, age_means, age_std):
    demographic = np.asmatrix(demographic)
    demographic[:, 0] = (demographic[:, 0] - age_means) / age_std
    return demographic.tolist()
