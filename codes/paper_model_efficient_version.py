#%% Import modules
import os, re, sys
sys.path.append("D:/2_Project/KOHI/2_codes/original/")
os.environ["KERAS_BACKEND"]="tensorflow"
import numpy as np
import pandas as pd
from mimic3models.common_keras_models import lstm_cnn
from mimic3benchmark.util import *
from mimic3models.readmission import utils
from mimic3benchmark.readers import ReadmissionReader
from mimic3models.preprocessing import Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils
from keras.callbacks import ModelCheckpoint, CSVLogger
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
    diagnoses=diagnoses.ix[(diagnoses.ICUSTAY_ID==int(icustay))]
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
    insurance=insurance.ix[(insurance.ICUSTAY_ID==int(icustay))]
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
    max_len = max([x.shape[0] for x in arr])
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret)

#%% Set hyper-parameters
l1 = 0
l2 = 0
dim = 16
depth = 2
epochs= 50
dropout = 0.3
batch_size = 8
timestep = 1.0
target_repl =0.0
rec_dropout = 0.0
target_repl_coef = 0.0
task = 'ihm'
prefix = ""
network = 'lstm_cnn'
loss = 'binary_crossentropy'
loss_weights = None

#%% Set dirs
dataset_dir    ='D:/2_Project/KOHI/1_data/2_created_readmission_data/'
train_list_dir ='D:/2_Project/KOHI/1_data/3_train_val_test/0_train_listfile801010.csv'
test_list_dir  ='D:/2_Project/KOHI/1_data/3_train_val_test/0_test_listfile801010.csv'
val_list_dir   ='D:/2_Project/KOHI/1_data/3_train_val_test/0_val_listfile801010.csv'
output_path    = 'D:/2_Project/KOHI/4_results/2_paper_model'

#%% Load Embeddings, Discretizer and settings for normalizer
age_means = 93.0
age_std = 23.99553529900332
cont_channels = [2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
normalizer = Normalizer(fields=cont_channels)  # choose here onlycont vs all
normalizer.load_params('D:/2_Project/KOHI/1_data/pp.pickle')
embeddings, word_indices = get_embeddings(corpus='claims_codes_hs', dim=300)
discretizer = Discretizer(timestep=float(timestep), store_masks=True,  imput_strategy='previous', start_time='zero')






#%% 기존 구성
train_list = open(train_list_dir, "r").readlines()
listfile_header = train_list[0]  # 'stay,period_length,y_true
data = train_list[1:]
data = [line.split(',') for line in data]  # ['29530_295275_episode1_timeseries_readmission.csv', '55.1976', '0\n']
data = [(x, float(t), int(y)) for (x, t, y) in data]  # ('48777_277581_episode1_timeseries_readmission.csv', 145.1856, 0)


#%% pandas로 수정
train_list = pd.read_csv(train_list_dir)
listfile_header = train_list.columns.values  # array(['stay', 'period_length', 'y_true'], dtype=object)
names = train_list["stay"].values

# Label에 따라 데이터 index를 추림.
y_true_0_idx = train_list[train_list['y_true']==0].index # 0 label의 index
y_true_1_idx = train_list[train_list['y_true']==1].index # 1 label의 index
# 1 label 수만큼 0 label의 데이터에서 index를 sampling
sampled_y_true_0_idx = np.random.choice(y_true_0_idx, len(y_true_1_idx),replace=False)
# Train_list로부터 balanced samples를 추림.
sampled_train_list = train_list.loc[list(sampled_y_true_0_idx)+list(y_true_1_idx)]
# Samples를 shuffle하고 index를 reset.
shuffled_sampled_train_list = sampled_train_list.sample(frac=1).reset_index(drop=True)
print('=== {} balanced samples are shuffled ==='.format(len(shuffled_sampled_train_list)))

order = list(shuffled_sampled_train_list.index)
groups = [[order[x % len(order)] for x in range(i, i + batch_size)] for i in range(0, len(order), batch_size)]

#TODO: 그룹 불러들이는 부분 구현해야 함.
group = groups[0]
# batch dataset load
batch_dict = {}
for i in group:
    temp_train_list = shuffled_sampled_train_list.loc[i] # csv, los, y_true
    fname = temp_train_list['stay']
    t = temp_train_list['period_length']
    y = temp_train_list['y_true']
    ret = read_example(fname, t, y)
    for k, v in ret.items(): # k -> "X", "t", "y", "header", "name"
        if k not in batch_dict:
            batch_dict[k] = []
        batch_dict[k].append(v)
batch_dict["header"] = batch_dict["header"][0]

data = batch_dict["X"]
ts = batch_dict["t"]
data = [discretizer.transform_end_t_hours(X, los=t)[0] for (X, t) in zip(data, ts)]

labels = batch_dict["y"]
names = batch_dict["name"]
diseases_list=get_diseases(names, 'D:/2_Project/KOHI/1_data/1_preprocessed/')
diseases_embedding=disease_embedding(embeddings, word_indices, diseases_list)
demographic=get_demographic(names, 'D:/2_Project/KOHI/1_data/1_preprocessed/')
demographic=age_normalize(demographic, age_means, age_std)
if (normalizer is not None):
    data = [normalizer.transform(X) for X in data]
data = [np.hstack([X, [d] * len(X)]) for (X, d) in zip(data, diseases_embedding)]
data = [np.hstack([X, [d] * len(X)]) for (X, d) in zip(data, demographic)]
pad_data = pad_zeros(data)
train_raw = (pad_data, np.array(labels))


print('train_raw: ', len(train_raw[0]))
print('train_raw train_raw[0][0]: ', len(train_raw[0][0]))
print('train_raw train_raw[0][1]: ', len(train_raw[0][1]))
print('train_raw: ', len(train_raw[0][0][0]))

# Build the model
print ("==> using model {}".format(network))
model = lstm_cnn.Network(dim=dim, batch_norm = True, dropout = dropout, rec_dropout = rec_dropout, task = task,
                target_repl=target_repl, deep_supervision=False, num_classes=1,
                depth=depth, input_dim=390,)
suffix = ".bs{}{}{}.ts{}{}".format(2, ".L1{}".format(l1) if l1 > 0 else "", ".L2{}".format(l2) if l2 > 0 else "",
                                   timestep, ".trc{}".format(target_repl_coef) if target_repl_coef > 0 else "")
model.final_name = prefix + model.say_name() + suffix
print ("==> model.final_name:", model.final_name)
print ("==> compiling the model")
print(model)
model.compile(optimizer=Adam(lr=0.001, beta_1=0.9), loss=loss,loss_weights=loss_weights)
model.summary()



#%% Load val data
val_list = pd.read_csv(val_list_dir)
val_names = val_list["stay"].values
val_dict = {}
for i in range(20): #TODO: 전체 데이터 불러들이는 부분 구현해야 함.
    temp_val_list = val_list.loc[i] # csv, los, y_true
    fname = temp_val_list['stay']
    t = temp_val_list['period_length']
    y = temp_val_list['y_true']
    ret = read_example(fname, t, y)
    for k, v in ret.items(): # k -> "X", "t", "y", "header", "name"
        if k not in val_dict:
            val_dict[k] = []
        val_dict[k].append(v)
val_dict["header"] = val_dict["header"][0]

val_data = val_dict["X"]
val_ts = val_dict["t"]
val_data = [discretizer.transform_end_t_hours(X, los=t)[0] for (X, t) in zip(val_data, val_ts)]

val_labels = val_dict["y"]
val_names = val_dict["name"]
val_diseases_list=get_diseases(val_names, 'D:/2_Project/KOHI/1_data/1_preprocessed/')
val_diseases_embedding=disease_embedding(embeddings, word_indices, val_diseases_list)
val_demographic=get_demographic(val_names, 'D:/2_Project/KOHI/1_data/1_preprocessed/')
val_demographic=age_normalize(val_demographic, age_means, age_std)

if (normalizer is not None):
    val_data = [normalizer.transform(X) for X in val_data]
val_data = [np.hstack([X, [d] * len(X)]) for (X, d) in zip(val_data,val_diseases_embedding)]
val_data = [np.hstack([X, [d] * len(X)]) for (X, d) in zip(val_data, val_demographic)]
val_pad_data = pad_zeros(val_data)
val_raw = (val_pad_data, np.array(val_labels))






# Prepare training
path = output_path+'/keras_state/' + model.final_name + '.epoch{epoch}.test{val_loss}.state'

metrics_callback = keras_utils.ReadmissionMetrics(train_data=train_raw,
                                                          val_data=val_raw,
                                                          target_repl=(target_repl_coef > 0),
                                                          batch_size=2,
                                                          verbose=2)
# make sure save directory exists
dirname = os.path.dirname(path)
if not os.path.exists(dirname):
    os.makedirs(dirname)
saver = ModelCheckpoint(path, verbose=1, period=20)

if not os.path.exists('keras_logs'):
    os.makedirs('keras_logs')
csv_logger = CSVLogger(os.path.join('keras_logs', model.final_name + '.csv'),
                       append=True, separator=';')

print ("==> training")
model.fit(x=train_raw[0],
          y=train_raw[1],
          validation_data=val_raw,
          nb_epoch=epochs,
          callbacks=[metrics_callback, saver, csv_logger],
          shuffle=True,
          verbose=2,
          batch_size=2)




#%% test
test_list = pd.read_csv(test_list_dir)
test_names = test_list["stay"].values
test_dict = {}
for i in range(20): #TODO: 전체 데이터 불러들이는 부분 구현해야 함.
    temp_test_list = test_list.loc[i] # csv, los, y_true
    fname = temp_test_list['stay']
    t = temp_test_list['period_length']
    y = temp_test_list['y_true']
    ret = read_example(fname, t, y)
    for k, v in ret.items(): # k -> "X", "t", "y", "header", "name"
        if k not in test_dict:
            test_dict[k] = []
        test_dict[k].append(v)
test_dict["header"] = test_dict["header"][0]

test_data = test_dict["X"]
test_ts = test_dict["t"]
test_data = [discretizer.transform_end_t_hours(X, los=t)[0] for (X, t) in zip(test_data, test_ts)]

test_labels = test_dict["y"]
test_names = test_dict["name"]
test_diseases_list=get_diseases(test_names, 'D:/2_Project/KOHI/1_data/1_preprocessed/')
test_diseases_embedding=disease_embedding(embeddings, word_indices, test_diseases_list)
test_demographic=get_demographic(test_names, 'D:/2_Project/KOHI/1_data/1_preprocessed/')
test_demographic=age_normalize(test_demographic, age_means, age_std)

if (normalizer is not None):
    test_data = [normalizer.transform(X) for X in test_data]
test_data = [np.hstack([X, [d] * len(X)]) for (X, d) in zip(test_data,test_diseases_embedding)]
test_data = [np.hstack([X, [d] * len(X)]) for (X, d) in zip(test_data, test_demographic)]
test_pad_data = pad_zeros(test_data)
test_raw = (test_pad_data, np.array(test_labels))

predictions = model.predict(test_pad_data, batch_size=1, verbose=1)
predictions = np.array(predictions)[:, 0]
metrics.print_metrics_binary(test_labels, predictions)


path = os.path.join(output_path,"test_predictions.csv")
utils.save_results(test_names, predictions, test_labels, path)

