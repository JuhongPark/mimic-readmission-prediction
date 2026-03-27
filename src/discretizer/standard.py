import numpy as np


class StandardDiscretizer:
    """17-channel discretizer for model training scripts."""

    def __init__(self, timestep=0.8, store_masks=True, impute_strategy='zero', start_time='zero'):
        self._id_to_channel = [
            'Capillary refill rate',
            'Diastolic blood pressure',
            'Fraction inspired oxygen',
            'Glasgow coma scale eye opening',
            'Glasgow coma scale motor response',
            'Glasgow coma scale total',
            'Glasgow coma scale verbal response',
            'Glucose',
            'Heart Rate',
            'Height',
            'Mean blood pressure',
            'Oxygen saturation',
            'Respiratory rate',
            'Systolic blood pressure',
            'Temperature',
            'Weight',
            'pH',
        ]
        self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))
        self._is_categorical_channel = {
            'Capillary refill rate': True,
            'Diastolic blood pressure': False,
            'Fraction inspired oxygen': False,
            'Glasgow coma scale eye opening': True,
            'Glasgow coma scale motor response': True,
            'Glasgow coma scale total': True,
            'Glasgow coma scale verbal response': True,
            'Glucose': False,
            'Heart Rate': False,
            'Height': False,
            'Mean blood pressure': False,
            'Oxygen saturation': False,
            'Respiratory rate': False,
            'Systolic blood pressure': False,
            'Temperature': False,
            'Weight': False,
            'pH': False,
        }
        self._possible_values = {
            'Capillary refill rate': ['0.0', '1.0'],
            'Diastolic blood pressure': [],
            'Fraction inspired oxygen': [],
            'Glasgow coma scale eye opening': [
                'To Pain', '3 To speech', '1 No Response', '4 Spontaneously',
                'None', 'To Speech', 'Spontaneously', '2 To pain',
            ],
            'Glasgow coma scale motor response': [
                '1 No Response', '3 Abnorm flexion', 'Abnormal extension',
                'No response', '4 Flex-withdraws', 'Localizes Pain',
                'Flex-withdraws', 'Obeys Commands', 'Abnormal Flexion',
                '6 Obeys Commands', '5 Localizes Pain', '2 Abnorm extensn',
            ],
            'Glasgow coma scale total': [
                '11', '10', '13', '12', '15', '14', '3', '5', '4', '7', '6', '9', '8',
            ],
            'Glasgow coma scale verbal response': [
                '1 No Response', 'No Response', 'Confused', 'Inappropriate Words',
                'Oriented', 'No Response-ETT', '5 Oriented',
                'Incomprehensible sounds', '1.0 ET/Trach', '4 Confused',
                '2 Incomp sounds', '3 Inapprop words',
            ],
            'Glucose': [],
            'Heart Rate': [],
            'Height': [],
            'Mean blood pressure': [],
            'Oxygen saturation': [],
            'Respiratory rate': [],
            'Systolic blood pressure': [],
            'Temperature': [],
            'Weight': [],
            'pH': [],
        }
        self._normal_values = {
            'Capillary refill rate': '0.0',
            'Diastolic blood pressure': '59.0',
            'Fraction inspired oxygen': '0.21',
            'Glasgow coma scale eye opening': '4 Spontaneously',
            'Glasgow coma scale motor response': '6 Obeys Commands',
            'Glasgow coma scale total': '15',
            'Glasgow coma scale verbal response': '5 Oriented',
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
        self._impute_strategy = impute_strategy
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0
        self._missing_data = 0
        self._missing_data_proposition = 0
        self._stay_with_missing_data = 0

    # Full 65-column original header used for index mapping
    _ORI_HEADER = [
        'Hours', 'Alanine aminotransferase', 'Albumin', 'Alkaline phosphate',
        'Anion gap', 'Asparate aminotransferase', 'Basophils', 'Bicarbonate',
        'Bilirubin', 'Blood culture', 'Blood urea nitrogen', 'Calcium',
        'Calcium ionized', 'Capillary refill rate', 'Chloride', 'Cholesterol',
        'Creatinine', 'Diastolic blood pressure', 'Eosinophils',
        'Fraction inspired oxygen', 'Glasgow coma scale eye opening',
        'Glasgow coma scale motor response', 'Glasgow coma scale total',
        'Glasgow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height',
        'Hematocrit', 'Hemoglobin', 'Lactate', 'Lactate dehydrogenase',
        'Lactic acid', 'Lymphocytes', 'Magnesium', 'Mean blood pressure',
        'Mean corpuscular hemoglobin', 'Mean corpuscular hemoglobin concentration',
        'Mean corpuscular volume', 'Monocytes', 'Neutrophils', 'Oxygen saturation',
        'Partial pressure of carbon dioxide', 'Partial pressure of oxygen',
        'Partial thromboplastin time', 'Peak inspiratory pressure', 'Phosphate',
        'Platelets', 'Positive end-expiratory pressure', 'Potassium',
        'Prothrombin time', 'Pupillary response left', 'Pupillary response right',
        'Pupillary size left', 'Pupillary size right', 'Red blood cell count',
        'Respiratory rate', 'Sodium', 'Systolic blood pressure', 'Temperature',
        'Troponin-I', 'Troponin-T', 'Urine output', 'Weight',
        'White blood cell count', 'pH',
    ]

    def transform_end_t_hours(self, X, header=None, los=None, max_length=48):
        if header is None:
            header = self._header
        assert header[0] == "Hours"

        oriheader = self._ORI_HEADER
        indexbox = [oriheader.index(x) for x in header if x in oriheader]

        eps = 1e-6
        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i + 1] + eps

        if los > max_length:
            max_hours = max_length
            first_time = los - max_length
        else:
            max_hours = los
            first_time = 0

        N_bins = int(max_hours / self._timestep + 1.0 - eps)
        cur_len = 0
        begin_pos = [0] * N_channels
        end_pos = [0] * N_channels
        for i in range(N_channels):
            channel = self._id_to_channel[i]
            begin_pos[i] = cur_len
            if self._is_categorical_channel[channel]:
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
            if self._is_categorical_channel[channel]:
                category_id = self._possible_values[channel].index(value)
                N_values = len(self._possible_values[channel])
                one_hot = np.zeros((N_values,))
                one_hot[category_id] = 1
                for pos in range(N_values):
                    data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos]
            else:
                data[bin_id, begin_pos[channel_id]] = float(value)

        for row in X:
            t = float(row[0]) - first_time
            if t < 0:
                continue
            bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins

            for j in range(1, len(row)):
                if row[j] == "" or j not in indexbox:
                    continue
                savej = j
                j = header.index(oriheader[j])
                channel = header[j]
                channel_id = self._channel_to_id[channel]
                j = savej
                total_data += 1
                if mask[bin_id][channel_id] == 1:
                    unused_data += 1
                mask[bin_id][channel_id] = 1
                write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]

        # impute missing values
        if self._impute_strategy not in ['zero', 'normal_value', 'previous', 'next']:
            raise ValueError("impute strategy is invalid")

        if self._impute_strategy in ['normal_value', 'previous']:
            prev_values = [[] for _ in range(N_channels)]
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if self._impute_strategy == 'normal_value':
                        imputed_value = self._normal_values[channel]
                    if self._impute_strategy == 'previous':
                        if len(prev_values[channel_id]) == 0:
                            imputed_value = self._normal_values[channel]
                        else:
                            imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        if self._impute_strategy == 'next':
            prev_values = [[] for _ in range(N_channels)]
            for bin_id in range(N_bins - 1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if len(prev_values[channel_id]) == 0:
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])

        new_header = []
        for channel in self._id_to_channel:
            if self._is_categorical_channel[channel]:
                for value in self._possible_values[channel]:
                    new_header.append(channel + "->" + value)
            else:
                new_header.append(channel)
        if self._store_masks:
            for channel in self._id_to_channel:
                new_header.append("mask->" + channel)

        new_header = ",".join(new_header)
        return (data, new_header)
