import numpy as np
import pandas as pd
import keras

from src.data.reader import read_example, get_diseases, get_demographic
from src.data.features import disease_embedding, age_normalize, get_wordvectors
from src.data.padding import pad_zeros


class Generator(keras.utils.Sequence):
    def __init__(
        self,
        mode=None,
        data_dir=None,
        batch_size=1,
        age_means=93.0,
        age_std=23.99553529900332,
        normalizer=None,
        discretizer=None,
        embeddings=None,
        word_indices=None,
        discharge_wv=None,
    ):
        self.mode = mode
        self.batch_size = batch_size
        self.age_means = age_means
        self.age_std = age_std
        self.normalizer = normalizer
        self.discretizer = discretizer
        self.embeddings = embeddings
        self.word_indices = word_indices
        self.discharge_wv = discharge_wv

        # Load whole list
        self.whole_data_list = pd.read_csv(data_dir)
        self.whole_header = self.whole_data_list.columns.values
        self.whole_names = self.whole_data_list["stay"].values

        # Make balanced dataset for training
        if self.mode == 'train':
            self.data_list = self.balanced_sample_from_whole_list(self.whole_data_list)
        elif self.mode in ('val', 'test'):
            self.data_list = self.whole_data_list
        else:
            print('Please check your mode')

        # Make batch groups
        order = list(range(self.size()))
        self.groups = [
            [order[x % len(order)] for x in range(i, i + self.batch_size)]
            for i in range(0, len(order), self.batch_size)
        ]
        self.current_index = 0
        super(Generator, self).__init__()

    def balanced_sample_from_whole_list(self, whole_data_list):
        y_true_0_idx = whole_data_list[whole_data_list['y_true'] == 0].index
        y_true_1_idx = whole_data_list[whole_data_list['y_true'] == 1].index
        sampled_y_true_0_idx = np.random.choice(y_true_0_idx, len(y_true_1_idx), replace=False)
        sampled_data_list = whole_data_list.loc[list(sampled_y_true_0_idx) + list(y_true_1_idx)]
        shuffled_sampled_data_list = sampled_data_list.sample(frac=1).reset_index(drop=True)
        print('=== {} balanced samples are shuffled ==='.format(len(shuffled_sampled_data_list)))
        return shuffled_sampled_data_list

    def size(self):
        return len(self.data_list)

    def compute_inputs_targets(self, group):
        batch_dict = {}
        for i in range(len(group)):
            temp_train_list = self.data_list.loc[group[i]]
            fname = temp_train_list['stay']
            t = temp_train_list['period_length']
            y = temp_train_list['y_true']
            ret = read_example(fname, t, y)
            for k, v in ret.items():
                if k not in batch_dict:
                    batch_dict[k] = []
                batch_dict[k].append(v)

        ts = batch_dict["t"]
        data = batch_dict["X"]
        labels = batch_dict["y"]
        names = batch_dict["name"]

        data = [self.discretizer.transform_end_t_hours(X, los=t)[0] for (X, t) in zip(data, ts)]

        diseases_list = get_diseases(names)
        diseases_emb = disease_embedding(self.embeddings, self.word_indices, diseases_list)
        demographic = get_demographic(names)
        demographic = age_normalize(demographic, self.age_means, self.age_std)

        if self.normalizer is not None:
            data = [self.normalizer.transform(X) for X in data]
        data = [np.hstack([X, [d] * len(X)]) for (X, d) in zip(data, diseases_emb)]
        data = [np.hstack([X, [d] * len(X)]) for (X, d) in zip(data, demographic)]

        if self.discharge_wv is not None:
            wordvectors = get_wordvectors(names, self.discharge_wv)
            data = [np.hstack([X, [d] * len(X)]) for (X, d) in zip(data, wordvectors)]

        pad_data = pad_zeros(data)
        return pad_data, np.array(labels)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        if self.current_index >= len(self.groups):
            self.current_index = self.current_index % len(self.groups)
            if self.mode == 'train':
                self.data_list = self.balanced_sample_from_whole_list(self.whole_data_list)
                order = list(range(self.size()))
                self.groups = [
                    [order[x % len(order)] for x in range(i, i + self.batch_size)]
                    for i in range(0, len(order), self.batch_size)
                ]
        group = self.groups[self.current_index]
        inputs, targets = self.compute_inputs_targets(group)
        self.current_index = self.current_index + 1
        return inputs, targets
