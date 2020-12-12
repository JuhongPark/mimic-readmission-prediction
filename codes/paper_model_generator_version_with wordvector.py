#%% Import modules
import os, re, sys, keras
sys.path.append("D:/2_Project/KOHI/2_codes/original/")
os.environ["KERAS_BACKEND"]="tensorflow"
import numpy as np
import pandas as pd
import pickle as pickle
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn import metrics
from mimic3models.common_keras_models import lstm_cnn
from mimic3benchmark.util import *
from mimic3models.preprocessing import Normalizer
from keras.optimizers import Adam
from utilities.data_loader import get_embeddings

#%% Define functions
g_map = { 'F': 1, 'M': 2 }
e_map = { 'ASIAN': 1,
          'BLACK': 2,
          'HISPANIC': 3,
          'WHITE': 4,
          'OTHER': 5, # map everything else to 5 (OTHER)
          'UNABLE TO OBTAIN': 0,
          'PATIENT DECLINED TO ANSWER': 0,
          'UNKNOWN': 0,
          '': 0 }
i_map={'Government': 0,
       'Self Pay': 1,
       'Medicare':2,
       'Private':3,
       'Medicaid':4}

class Discretizer():
    def __init__(self, timestep=0.8, store_masks=True, imput_strategy='zero', start_time='zero'):
        self._id_to_channel = [
            'Capillary refill rate',
            'Diastolic blood pressure',
            'Fraction inspired oxygen',
            'Glascow coma scale eye opening',
            'Glascow coma scale motor response',
            'Glascow coma scale total',
            'Glascow coma scale verbal response',
            'Glucose',
            'Heart Rate',
            'Height',
            'Mean blood pressure',
            'Oxygen saturation',
            'Respiratory rate',
            'Systolic blood pressure',
            'Temperature',
            'Weight',
            'pH']
        self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))
        self._is_categorical_channel = {
            'Capillary refill rate': True,
            'Diastolic blood pressure': False,
            'Fraction inspired oxygen': False,
            'Glascow coma scale eye opening': True,
            'Glascow coma scale motor response': True,
            'Glascow coma scale total': True,
            'Glascow coma scale verbal response': True,
            'Glucose': False,
            'Heart Rate': False,
            'Height': False,
            'Mean blood pressure': False,
            'Oxygen saturation': False,
            'Respiratory rate': False,
            'Systolic blood pressure': False,
            'Temperature': False,
            'Weight': False,
            'pH': False}
        self._possible_values = {
            'Capillary refill rate': ['0.0', '1.0'],
            'Diastolic blood pressure': [],
            'Fraction inspired oxygen': [],
            'Glascow coma scale eye opening': ['To Pain',
                                               '3 To speech',
                                               '1 No Response',
                                               '4 Spontaneously',
                                               'None',
                                               'To Speech',
                                               'Spontaneously',
                                               '2 To pain'],
            'Glascow coma scale motor response': ['1 No Response',
                                                  '3 Abnorm flexion',
                                                  'Abnormal extension',
                                                  'No response',
                                                  '4 Flex-withdraws',
                                                  'Localizes Pain',
                                                  'Flex-withdraws',
                                                  'Obeys Commands',
                                                  'Abnormal Flexion',
                                                  '6 Obeys Commands',
                                                  '5 Localizes Pain',
                                                  '2 Abnorm extensn'],
            'Glascow coma scale total': ['11', '10', '13', '12', '15', '14', '3', '5', '4', '7', '6', '9', '8'],
            'Glascow coma scale verbal response': ['1 No Response',
                                                   'No Response',
                                                   'Confused',
                                                   'Inappropriate Words',
                                                   'Oriented',
                                                   'No Response-ETT',
                                                   '5 Oriented',
                                                   'Incomprehensible sounds',
                                                   '1.0 ET/Trach',
                                                   '4 Confused',
                                                   '2 Incomp sounds',
                                                   '3 Inapprop words'],
            'Glucose': [],
            'Heart Rate': [],
            'Height': [],
            'Mean blood pressure': [],
            'Oxygen saturation': [],
            'Respiratory rate': [],
            'Systolic blood pressure': [],
            'Temperature': [],
            'Weight': [],
            'pH': []
        }
        self._normal_values = {
            'Capillary refill rate': '0.0',
            'Diastolic blood pressure': '59.0',
            'Fraction inspired oxygen': '0.21',
            'Glascow coma scale eye opening': '4 Spontaneously',
            'Glascow coma scale motor response': '6 Obeys Commands',
            'Glascow coma scale total': '15',
            'Glascow coma scale verbal response': '5 Oriented',
            'Glucose': '128.0',
            'Heart Rate': '86',
            'Height': '170.0',
            'Mean blood pressure': '77.0',
            'Oxygen saturation': '98.0',
            'Respiratory rate': '19',
            'Systolic blood pressure': '118.0',
            'Temperature': '36.6',
            'Weight': '81.0',
            'pH': '7.4',
        }
        self._header = ["Hours"] + self._id_to_channel
        self._timestep = timestep
        self._store_masks = store_masks
        self._start_time = start_time
        self._imput_strategy = imput_strategy
        # for statistics
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0
        self._missing_data = 0
        self._missing_data_proposition = 0
        self._stay_with_missing_data = 0
    def transform_end_t_hours(self, X, header=None, los=None, max_length=48):
        max_length=max_length
        if (header == None):
            header = self._header
        assert header[0] == "Hours"
        indexbox = []
        oriheader = ['Hours', 'Alanine aminotransferase', 'Albumin', 'Alkaline phosphate', 'Anion gap', 'Asparate aminotransferase', 'Basophils', 'Bicarbonate', 'Bilirubin', 'Blood culture', 'Blood urea nitrogen', 'Calcium', 'Calcium ionized', 'Capillary refill rate', 'Chloride', 'Cholesterol', 'Creatinine', 'Diastolic blood pressure', 'Eosinophils', 'Fraction inspired oxygen', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height', 'Hematocrit', 'Hemoglobin', 'Lactate', 'Lactate dehydrogenase', 'Lactic acid', 'Lymphocytes', 'Magnesium', 'Mean blood pressure', 'Mean corpuscular hemoglobin', 'Mean corpuscular hemoglobin concentration', 'Mean corpuscular volume', 'Monocytes', 'Neutrophils', 'Oxygen saturation', 'Partial pressure of carbon dioxide', 'Partial pressure of oxygen', 'Partial thromboplastin time', 'Peak inspiratory pressure', 'Phosphate', 'Platelets', 'Positive end-expiratory pressure', 'Potassium', 'Prothrombin time', 'Pupillary response left', 'Pupillary response right', 'Pupillary size left', 'Pupillary size right', 'Red blood cell count', 'Respiratory rate', 'Sodium', 'Systolic blood pressure', 'Temperature', 'Troponin-I', 'Troponin-T', 'Urine output', 'Weight', 'White blood cell count', 'pH']
        for x in header:
            if x in oriheader:
                indexbox.append(oriheader.index(x))
        eps = 1e-6
        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i + 1] + eps
        if los>max_length:
            max_hours = max_length
            first_time = los - max_length
        else:
            max_hours=los
            first_time = 0

        N_bins = int(max_hours / self._timestep + 1.0 - eps)
        cur_len = 0
        begin_pos = [0 for i in range(N_channels)]
        end_pos = [0 for i in range(N_channels)]
        for i in range(N_channels):
            channel = self._id_to_channel[i]
            begin_pos[i] = cur_len
            if (self._is_categorical_channel[channel]):
                end_pos[i] = begin_pos[i] + len(self._possible_values[channel])
            else:
                end_pos[i] = begin_pos[i] + 1
            cur_len = end_pos[i]

        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0

        def write(data, bin_id, channel, value, begin_pos):
            channel_id = self._channel_to_id[channel]
            if (self._is_categorical_channel[channel]):
                category_id = self._possible_values[channel].index(value)
                N_values = len(self._possible_values[channel])
                one_hot = np.zeros((N_values,))
                one_hot[category_id] = 1
                for pos in range(N_values):
                    data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos]
            else:
                data[bin_id, begin_pos[channel_id]] = float(value)

        for row in X:
            t = float(row[0])- first_time
            if (t < 0):
                continue
            bin_id = int(t / self._timestep - eps)
            assert (bin_id >= 0 and bin_id < N_bins)

            for j in range(1, len(row)):
                if row[j] == "" or j not in indexbox:
                    continue
                savej = j
                j = header.index(oriheader[j])
                channel = header[j]
                channel_id = self._channel_to_id[channel]
                j = savej
                total_data += 1
                if (mask[bin_id][channel_id] == 1):
                    unused_data += 1
                mask[bin_id][channel_id] = 1
                write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]
        # impute missing values
        if (self._imput_strategy not in ['zero', 'normal_value', 'previous', 'next']):
            raise ValueError("impute strategy is invalid")
        if (self._imput_strategy in ['normal_value', 'previous']):
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if (mask[bin_id][channel_id] == 1):
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if (self._imput_strategy == 'normal_value'):
                        imputed_value = self._normal_values[channel]
                    if (self._imput_strategy == 'previous'):
                        if (len(prev_values[channel_id]) == 0):
                            imputed_value = self._normal_values[channel]
                        else:
                            imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)
        if (self._imput_strategy == 'next'):
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins - 1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if (mask[bin_id][channel_id] == 1):
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if (len(prev_values[channel_id]) == 0):
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)
        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)
        if (self._store_masks):
            data = np.hstack([data, mask.astype(np.float32)])
        new_header = []
        for channel in self._id_to_channel:
            if (self._is_categorical_channel[channel]):
                values = self._possible_values[channel]
                for value in values:
                    new_header.append(channel + "->" + value)
            else:
                new_header.append(channel)
        if (self._store_masks):
            for i in range(len(self._id_to_channel)):
                channel = self._id_to_channel[i]
                new_header.append("mask->" + channel)
        new_header = ",".join(new_header)
        return (data, new_header)

def read_example(fname, t, y):
    ret = []
    with open(os.path.join(dataset_dir,fname), "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            # mas = line.strip().split(',')
            mas = line.strip()
            mas = re.sub(r'("[^"]*")|,', lambda x: x.group(1).replace(',', '') if x.group(1) else x.group(), mas)
            mas = mas.split(',')
            ret.append(np.array(mas))
    X =  np.stack(ret)
    return {"X": X, "t": t, "y": y, "header": header, "name": fname}

def read_diagnose(subject_path,icustay):
    diagnoses = dataframe_from_csv(os.path.join(subject_path, 'diagnoses.csv'), index_col=None)
    diagnoses=diagnoses.loc[(diagnoses.ICUSTAY_ID==int(icustay))]
    diagnoses=diagnoses['ICD9_CODE'].values.tolist()
    return diagnoses

def get_diseases(names,path):
    disease_list=[]
    namelist=[]
    for element in names:
        x=element.split('_')
        namelist.append((x[0],x[2]))
    for x in namelist:
        subject=x[0]
        icustay=x[1]
        subject_path=os.path.join(path, subject)
        disease = read_diagnose(subject_path,icustay)
        disease_list.append(disease)
    return disease_list

def get_wordvectors(names, discharge_wv):
    wordvector_list=[]
    namelist=[]
    for element in names:
        x=element.split('_')
        namelist.append(x[2])
    for icustay in namelist:
        wordvector = discharge_wv[discharge_wv.ICUSTAY_ID==icustay].k500_mean_wv #500 keyword 사용하는 경우
        if len(wordvector)!=200:
            wordvector = [0] * 200 # word 추출 없던 경우는 0 vector로만
            wordvector_list.append(wordvector)
        else:
            wordvector_list.append(wordvector[0].reshape((-1,1)))
    return wordvector_list

def read_demographic(subject_path,icustay,episode):
    demographic_re=[0]*14
    demographic = dataframe_from_csv(os.path.join(subject_path, episode+'_readmission.csv'), index_col=None)
    age_start=0
    gender_start=1
    enhnicity_strat=3
    insurance_strat=9
    demographic_re[age_start]=float(demographic['Age'].iloc[0])
    demographic_re[gender_start-1+int(demographic['Gender'].iloc[0])]=1 # Female 1, Male 2, Other 3
    demographic_re[enhnicity_strat+int(demographic['Ethnicity'].iloc[0])]=1 # 5 Ethnicity
    insurance =dataframe_from_csv(os.path.join(subject_path, 'stays_readmission.csv'), index_col=None)
    insurance=insurance.loc[(insurance.ICUSTAY_ID==int(icustay))]
    demographic_re[insurance_strat+i_map[insurance['INSURANCE'].iloc[0]]]=1
    return demographic_re

def get_demographic(names,path):
    demographic_list=[]
    namelist=[]
    for element in names:
        x=element.split('_')
        namelist.append((x[0],x[2],x[3]))
    for x in namelist:
        subject=x[0]
        icustay=x[1]
        episode=x[2]
        subject_path=os.path.join(path, subject)
        demographic = read_demographic(subject_path,icustay,episode)
        demographic_list.append(demographic)
    return demographic_list

def disease_embedding(embeddings, word_indices,diseases_list):
    emb_list=[]
    for diseases in diseases_list:
        emb_period=[0]*300
        skip=0
        for disease in diseases:
            k='IDX_'+str(disease)
            if k not in word_indices.keys():
                skip+=1
                continue
            index=word_indices[k]
            emb_disease=embeddings[index]
            emb_period = [sum(x) for x in zip(emb_period, emb_disease)]
        emb_period = [x / len(diseases) for x in emb_period]
        emb_list.append(emb_period)
    return emb_list

def age_normalize(demographic, age_means, age_std):
    demographic = np.asmatrix(demographic)
    demographic[:,0] = (demographic[:,0] - age_means) / age_std
    return demographic.tolist()

def pad_zeros(arr, min_length=None):
    """
    `arr` is an array of `np.array`s

    The function appends zeros to every `np.array` in `arr`
    to equalize their first axis lenghts.
    """
    dtype = arr[0].dtype
    max_len = 48
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret)

def print_metrics_binary(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    if (len(predictions.shape) == 1):
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        print ("confusion matrix:")
        print (cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        print ("accuracy =", acc)
        print ("precision class 0 =", prec0)
        print ("precision class 1 =", prec1)
        print ("recall class 0 =", rec0)
        print ("recall calss 1 =", rec1)
        print ("AUC of ROC =", auroc)
        print ("AUC of PRC =", auprc)

    return {"acc": acc,
            "prec0": prec0,
            "prec1": prec1,
            "rec0": rec0,
            "rec1": rec1,
            "auroc": auroc,
            "auprc": auprc,
            "minpse": minpse}


#%% Set hyper-parameters
l1 = 0
l2 = 0
dim = 16
depth = 2
epochs= 50
dropout = 0.3
batch_size = 100
timestep = 1.0
target_repl =0.0
rec_dropout = 0.0
target_repl_coef = 0.0
task = 'ihm'
network = 'lstm_cnn'
loss = 'binary_crossentropy'
loss_weights = None
experiment = 'experiment5'

#%% Set dirs
dataset_dir    ='D:/2_Project/KOHI/1_data/2_created_readmission_data/'
train_list_dir ='D:/2_Project/KOHI/1_data/3_train_val_test/0_train_listfile801010.csv'
test_list_dir  ='D:/2_Project/KOHI/1_data/3_train_val_test/0_test_listfile801010.csv'
val_list_dir   ='D:/2_Project/KOHI/1_data/3_train_val_test/0_val_listfile801010.csv'
output_path    = os.path.join('D:/2_Project/KOHI/4_results/2_paper_model', experiment)

#%% Load Embeddings, Discretizer and settings for normalizer
age_means = 93.0
age_std = 23.99553529900332
cont_channels = [2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
normalizer = Normalizer(fields=cont_channels)  # choose here onlycont vs all
normalizer.load_params('D:/2_Project/KOHI/1_data/normalizer.pickle') # 전체 데이터로부터 normalize 한 값들을 pickle로 저장하고, 불러오셔야 합니다.
embeddings, word_indices = get_embeddings(corpus='claims_codes_hs', dim=300)
discretizer = Discretizer(timestep=float(timestep), store_masks=True,  imput_strategy='previous', start_time='zero')

norm_dist_list = [0] * 300
for index in range(len(norm_dist_list)):
    index_data = embeddings[:, index]
    dist = norm(index_data.mean(), index_data.std())
    norm_dist_list[index] = dist

discharge_wv= pd.read_pickle('D:/2_Project/KOHI/1_data/keyword_discharge_note_wv_v3.pk') # word vector 저장한 pickle dir 부르는 부분

#%% Generator
class Generator(keras.utils.Sequence):
    def __init__(
            self,
            mode        =None,
            data_dir    =None,
            batch_size  = 1,
            age_means   = 93.0,
            age_std     = 23.99553529900332,
            normalizer  = None,
            discretizer = None,
            embeddings  = None,
            word_indices= None,
            discharge_wv= None,
    ):
        self.mode        = mode
        self.batch_size  = batch_size
        self.age_means   = age_means
        self.age_std     = age_std
        self.normalizer  = normalizer
        self.discretizer = discretizer
        self.embeddings  = embeddings
        self.word_indices= word_indices
        self.discharge_wv=discharge_wv
        # Load whole list
        self.whole_data_list = pd.read_csv(data_dir)
        self.whole_header = self.whole_data_list.columns.values # array(['stay','period_length','y_true'], dtype=object)
        self.whole_names = self.whole_data_list["stay"].values
        # Make balanced dataset for training
        if self.mode == 'train':
            self.data_list = self.balanced_sample_from_whole_list(self.whole_data_list)
        elif self.mode == 'val' or self.mode=='test':
            self.data_list = self.whole_data_list
        else :
            print('Please check your mode')
        # Make batch groups
        order = list(range(self.size()))
        self.groups = [[order[x % len(order)] for x in range(i, i+self.batch_size)]
                       for i in range(0, len(order), self.batch_size)]
        self.current_index=0
        super(Generator, self).__init__()

    def balanced_sample_from_whole_list(self, whole_data_list):
        y_true_0_idx = whole_data_list[whole_data_list['y_true'] == 0].index  # 0 label의 index
        y_true_1_idx = whole_data_list[whole_data_list['y_true'] == 1].index  # 1 label의 index
        # 1 label 수만큼 0 label의 데이터에서 index를 sampling
        sampled_y_true_0_idx = np.random.choice(y_true_0_idx, len(y_true_1_idx), replace=False)
        # Train_list로부터 balanced samples를 추림.
        sampled_data_list = whole_data_list.loc[list(sampled_y_true_0_idx) + list(y_true_1_idx)]
        # Samples를 shuffle하고 index를 reset.
        shuffled_sampled_data_list = sampled_data_list.sample(frac=1).reset_index(drop=True)
        print('=== {} balanced samples are shuffled ==='.format(len(shuffled_sampled_data_list)))
        return shuffled_sampled_data_list

    # Define private internal functions
    def size(self):
        return len(self.data_list)

    def compute_inputs_targets(self, group):
        batch_dict = {}
        for i in range(len(group)):
            temp_train_list = self.data_list.loc[group[i]]  # csv, los, y_true
            fname = temp_train_list['stay']
            t = temp_train_list['period_length']
            y = temp_train_list['y_true']
            ret = read_example(fname, t, y)
            for k, v in ret.items():  # k -> "X", "t", "y", "header", "name"
                if k not in batch_dict:
                    batch_dict[k] = []
                batch_dict[k].append(v)

        ts = batch_dict["t"]
        data = batch_dict["X"]
        labels = batch_dict["y"]
        names = batch_dict["name"]

        data = [self.discretizer.transform_end_t_hours(X, los=t)[0] for (X, t) in zip(data, ts)]

        diseases_list = get_diseases(names, 'D:/2_Project/KOHI/1_data/1_preprocessed/')
        diseases_embedding = disease_embedding(self.embeddings, self.word_indices, diseases_list)
        demographic = get_demographic(names, 'D:/2_Project/KOHI/1_data/1_preprocessed/')
        demographic = age_normalize(demographic, self.age_means, self.age_std)
        wordvectors = get_wordvectors(names, self.discharge_wv)
        if (self.normalizer is not None):
            data = [self.normalizer.transform(X) for X in data]
        data = [np.hstack([X, [d] * len(X)]) for (X, d) in zip(data, diseases_embedding)]
        data = [np.hstack([X, [d] * len(X)]) for (X, d) in zip(data, demographic)]
        data = [np.hstack([X, [d] * len(X)]) for (X, d) in zip(data, wordvectors)]
        pad_data = pad_zeros(data)

        return pad_data, np.array(labels)

    # Define batch related internal functions
    def __len__(self):
        # Number of batches for generator.
        return len(self.groups)

    def __getitem__(self, index):
        # Keras sequence method for generating batches.
        if self.current_index >= len(self.groups):
            self.current_index = self.current_index % (len(self.groups))
            if self.mode == 'train':
                self.data_list = self.balanced_sample_from_whole_list(self.whole_data_list)
                order = list(range(self.size()))
                self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)]
                               for i in range(0, len(order), self.batch_size)]
        group = self.groups[self.current_index]
        inputs, targets = self.compute_inputs_targets(group)
        self.current_index = self.current_index + 1
        return inputs, targets

train_generator = Generator(mode='train', data_dir=train_list_dir, batch_size=batch_size, normalizer=normalizer,
                            discretizer=discretizer, embeddings=embeddings, word_indices=word_indices, discharge_wv=discharge_wv)

val_generator = Generator(mode='val', data_dir=val_list_dir, batch_size=1, normalizer=normalizer,
                          discretizer=discretizer, embeddings=embeddings, word_indices=word_indices, discharge_wv= discharge_wv)

print('train data size: ', train_generator.size())
print('validataion data size: ', val_generator.size())


#%% Build the model
print ("==> using model {}".format(network))
model = lstm_cnn.Network(dim=dim, batch_norm = True, dropout = dropout, rec_dropout = rec_dropout, task = task,
                target_repl=target_repl, deep_supervision=False, num_classes=1,
                depth=depth, input_dim=590,)
model.final_name = model.say_name()
print ("==> model.final_name:", model.final_name)
print ("==> compiling the model")
print(model)
model.compile(optimizer=Adam(lr=0.001, beta_1=0.9), loss=loss, metrics=['acc'], loss_weights=loss_weights)
model.summary()

#%% Prepare training
path = output_path+'/keras_state/'+model.final_name
dirname = os.path.dirname(path)
if not os.path.exists(dirname):
    os.makedirs(dirname)

checkpoint = keras.callbacks.ModelCheckpoint(filepath=path+'.hdf5', save_best_only=True, monitor='val_acc',
                                             save_weights_only=True, mode='max', period=1)

# checkpoint = ModelCheckpoint(path, verbose=1, period=1)
hist = model.fit_generator(verbose=1, epochs=epochs, steps_per_epoch=train_generator.size()//batch_size, validation_steps=val_generator.size(),
                           generator=train_generator, validation_data=val_generator, callbacks=[checkpoint])

h = hist.history
def plot(hist, epochnum, batchnum, savepath, fullname):
    train_loss = hist['loss']
    val_loss = hist['val_loss']
    acc = hist['acc']
    val_acc = hist['val_acc']
    plt.figure()
    epochs = np.arange(1, len(train_loss) + 1, 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.title('Loss, Model={}, Epochs={}, Batch={}'.format(fullname, epochnum, batchnum))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.show() # <- When we run this before 'savefig', blank image files are generated.
    plt.savefig('{}/{}_Loss.png'.format(savepath, fullname), format='png')

    plt.figure()
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.title('Accuracy, Model={}, Epochs={}, Batch={}'.format(fullname, epochnum, batchnum))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # plt.show()
    plt.savefig('{}/{}_Accuracy.png'.format(savepath, fullname), format='png')
plot(h, epochs, batch_size, output_path, model.final_name)

#%% test
model.load_weights(path +'.hdf5')
test_generator = Generator(mode='test', data_dir=test_list_dir, batch_size=1, normalizer=normalizer,discharge_wv=discharge_wv,
                            discretizer=discretizer, embeddings=embeddings, word_indices=word_indices)
predictions = model.predict_generator(test_generator, steps=test_generator.size())
predictions = np.array(predictions)[:, 0]
test_labels = test_generator.data_list['y_true']
test_names  = test_generator.whole_names
plt.clf()

fpr, tpr, thresh = metrics.roc_curve(test_labels, predictions)
auc = metrics.roc_auc_score(test_labels, predictions)
plt.plot(fpr, tpr, lw=2, label="%s = %0.4f"%(model.final_name ,auc))
plt.plot([0, 1], [0, 1], linestyle='--',lw=2, color='k')
plt.xlim([0., 1.])
plt.ylim([0., 1.])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_path,'ROC0.png'))
plt.clf()

results = print_metrics_binary(test_labels, predictions)
path = os.path.join(output_path,"test_predictions.csv")

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def save_results(results, names, pred, y_true, path):
    makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("acc, prec0, prec1, rec0, rec1, auroc, auprc, minpse\n")
        for key in results.keys():
            f.write("{},".format(results[key]))

        f.write("\nstay,prediction,y_true\n")
        for (name, x, y) in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))


save_results(results, test_names, predictions, test_labels, path)

