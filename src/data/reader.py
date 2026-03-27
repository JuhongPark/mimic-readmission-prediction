import os
import re
import numpy as np
import pandas as pd

from config.paths import DATASET_DIR, PREPROCESSED_DIR
from config.defaults import i_map


def read_example(fname, t, y, dataset_dir=None):
    if dataset_dir is None:
        dataset_dir = DATASET_DIR
    ret = []
    with open(os.path.join(dataset_dir, fname), "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip()
            mas = re.sub(
                r'("[^"]*")|,',
                lambda x: x.group(1).replace(',', '') if x.group(1) else x.group(),
                mas,
            )
            mas = mas.split(',')
            ret.append(np.array(mas))
    X = np.stack(ret)
    return {"X": X, "t": t, "y": y, "header": header, "name": fname}


def read_diagnose(subject_path, icustay):
    diagnoses = pd.read_csv(os.path.join(subject_path, 'diagnoses.csv'))
    diagnoses = diagnoses.loc[diagnoses.ICUSTAY_ID == int(icustay)]
    diagnoses = diagnoses['ICD9_CODE'].values.tolist()
    return diagnoses


def get_diseases(names, path=None):
    if path is None:
        path = PREPROCESSED_DIR
    disease_list = []
    namelist = []
    for element in names:
        x = element.split('_')
        namelist.append((x[0], x[2]))
    for x in namelist:
        subject = x[0]
        icustay = x[1]
        subject_path = os.path.join(path, subject)
        disease = read_diagnose(subject_path, icustay)
        disease_list.append(disease)
    return disease_list


def read_demographic(subject_path, icustay, episode):
    demographic_re = [0] * 14
    demographic = pd.read_csv(os.path.join(subject_path, episode + '_readmission.csv'))
    age_start = 0
    gender_start = 1
    ethnicity_start = 3
    insurance_start = 9
    demographic_re[age_start] = float(demographic['Age'].iloc[0])
    demographic_re[gender_start - 1 + int(demographic['Gender'].iloc[0])] = 1
    demographic_re[ethnicity_start + int(demographic['Ethnicity'].iloc[0])] = 1
    insurance = pd.read_csv(os.path.join(subject_path, 'stays_readmission.csv'))
    insurance = insurance.loc[insurance.ICUSTAY_ID == int(icustay)]
    demographic_re[insurance_start + i_map[insurance['INSURANCE'].iloc[0]]] = 1
    return demographic_re


def get_demographic(names, path=None):
    if path is None:
        path = PREPROCESSED_DIR
    demographic_list = []
    namelist = []
    for element in names:
        x = element.split('_')
        namelist.append((x[0], x[2], x[3]))
    for x in namelist:
        subject = x[0]
        icustay = x[1]
        episode = x[2]
        subject_path = os.path.join(path, subject)
        demographic = read_demographic(subject_path, icustay, episode)
        demographic_list.append(demographic)
    return demographic_list
