import re

import numpy as np


class ExtendedDiscretizer:
    """50-channel discretizer with extended clinical variables."""

    def __init__(self, timestep=0.8, store_masks=True, imput_strategy='zero', start_time='zero'):

        self._id_to_channel = [
            'Alanine aminotransferase',
            'Albumin',
            'Alkaline phosphate',
            'Anion gap',
            'Asparate aminotransferase',
            'Bicarbonate',
            'Bilirubin',
            'Blood urea nitrogen',
            'Calcium',
            'Calcium ionized',
            'Chloride',
            'Creatinine',
            'Diastolic blood pressure',
            'Fraction inspired oxygen',
            'Capillary refill rate',
            'Glascow coma scale eye opening',
            'Glascow coma scale motor response',
            'Glascow coma scale total',
            'Glascow coma scale verbal response',
            'Glucose',
            'Hematocrit',
            'Hemoglobin',
            'Heart Rate',
            'Height',
            'Lactate',
            'Lactate dehydrogenase',
            'Lactic acid',
            'Magnesium',
            'Mean blood pressure',
            'Mean corpuscular hemoglobin',
            'Mean corpuscular hemoglobin concentration',
            'Mean corpuscular volume',
            'Oxygen saturation',
            'Partial pressure of cabon dioxide',
            'Partial pressure of oxygen',
            'Partial thromboplastin time',
            'Peak inspiratory pressure',
            'Phosphate',
            'Platelets',
            'Positive end-expiratory pressure',
            'Potassium',
            'Prothrombin time',
            'Respiratory rate',
            'Sodium',
            'Systolic blood pressure',
            'Temperature',
            'Troponin-I',
            'Troponin-T',
            'Urine output',
            'Weight',
            'White blood cell count',
            'pH',
        ]

        self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))

        self._is_categorical_channel = {
            'Alanine aminotransferase': False,
            'Albumin': False,
            'Alkaline phosphate': False,
            'Anion gap': False,
            'Asparate aminotransferase': False,
            'Bicarbonate': False,
            'Bilirubin': False,
            'Blood urea nitrogen': False,
            'Calcium': False,
            'Calcium ionized': False,
            'Chloride': False,
            'Creatinine': False,
            'Diastolic blood pressure': False,
            'Fraction inspired oxygen': False,
            'Capillary refill rate': True,
            'Glascow coma scale eye opening': True,
            'Glascow coma scale motor response': True,
            'Glascow coma scale total': True,
            'Glascow coma scale verbal response': True,
            'Glucose': False,
            'Hematocrit': False,
            'Hemoglobin': False,
            'Heart Rate': False,
            'Height': False,
            'Lactate': False,
            'Lactate dehydrogenase': False,
            'Lactic acid': False,
            'Magnesium': False,
            'Mean blood pressure': False,
            'Mean corpuscular hemoglobin': False,
            'Mean corpuscular hemoglobin concentration': False,
            'Mean corpuscular volume': False,
            'Oxygen saturation': False,
            'Partial pressure of cabon dioxide': False,
            'Partial pressure of oxygen': False,
            'Partial thromboplastin time': False,
            'Peak inspiratory pressure': False,
            'Phosphate': False,
            'Platelets': False,
            'Positive end-expiratory pressure': False,
            'Potassium': False,
            'Prothrombin time': False,
            'Respiratory rate': False,
            'Sodium': False,
            'Systolic blood pressure': False,
            'Temperature': False,
            'Troponin-I': False,
            'Troponin-T': False,
            'Urine output': False,
            'Weight': False,
            'White blood cell count': False,
            'pH': False,
        }

        self._possible_values = {
            'Alanine aminotransferase': [],
            'Albumin': [],
            'Alkaline phosphate': [],
            'Anion gap': [],
            'Asparate aminotransferase': [],
            'Bicarbonate': [],
            'Bilirubin': [],
            'Blood urea nitrogen': [],
            'Calcium': [],
            'Calcium ionized': [],
            'Chloride': [],
            'Creatinine': [],
            'Diastolic blood pressure': [],
            'Fraction inspired oxygen': [],
            'Capillary refill rate': ['0.0', '1.0'],
            'Glascow coma scale eye opening': [
                'To Pain', '3 To speech', '1 No Response', '4 Spontaneously',
                'None', 'To Speech', 'Spontaneously', '2 To pain',
            ],
            'Glascow coma scale motor response': [
                '1 No Response', '3 Abnorm flexion', 'Abnormal extension',
                'No response', '4 Flex-withdraws', 'Localizes Pain',
                'Flex-withdraws', 'Obeys Commands', 'Abnormal Flexion',
                '6 Obeys Commands', '5 Localizes Pain', '2 Abnorm extensn',
            ],
            'Glascow coma scale total': [
                '11', '10', '13', '12', '15', '14', '3', '5', '4', '7', '6', '9', '8',
            ],
            'Glascow coma scale verbal response': [
                '1 No Response', 'No Response', 'Confused', 'Inappropriate Words',
                'Oriented', 'No Response-ETT', '5 Oriented',
                'Incomprehensible sounds', '1.0 ET/Trach', '4 Confused',
                '2 Incomp sounds', '3 Inapprop words',
            ],
            'Glucose': [],
            'Hematocrit': [],
            'Hemoglobin': [],
            'Heart Rate': [],
            'Height': [],
            'Lactate': [],
            'Lactate dehydrogenase': [],
            'Lactic acid': [],
            'Magnesium': [],
            'Mean blood pressure': [],
            'Mean corpuscular hemoglobin': [],
            'Mean corpuscular hemoglobin concentration': [],
            'Mean corpuscular volume': [],
            'Oxygen saturation': [],
            'Partial pressure of cabon dioxide': [],
            'Partial pressure of oxygen': [],
            'Partial thromboplastin time': [],
            'Peak inspiratory pressure': [],
            'Phosphate': [],
            'Platelets': [],
            'Positive end-expiratory pressure': [],
            'Potassium': [],
            'Prothrombin time': [],
            'Respiratory rate': [],
            'Sodium': [],
            'Systolic blood pressure': [],
            'Temperature': [],
            'Troponin-I': [],
            'Troponin-T': [],
            'Urine output': [],
            'Weight': [],
            'White blood cell count': [],
            'pH': [],
        }

        self._normal_values = {
            'Alanine aminotransferase': '33',
            'Albumin': '3.4',
            'Alkaline phosphate': '140',
            'Anion gap': '10',
            'Asparate aminotransferase': '41',
            'Bicarbonate': '25',
            'Bilirubin': '1.2',
            'Blood urea nitrogen': '20',
            'Calcium': '10.3',
            'Calcium ionized': '5.28',
            'Chloride': '94',
            'Creatinine': '1.21',
            'Diastolic blood pressure': '59.0',
            'Fraction inspired oxygen': '0.21',
            'Capillary refill rate': '0.0',
            'Glascow coma scale eye opening': '4 Spontaneously',
            'Glascow coma scale motor response': '6 Obeys Commands',
            'Glascow coma scale total': '15',
            'Glascow coma scale verbal response': '5 Oriented',
            'Glucose': '128.0',
            'Hematocrit': '35.5',
            'Hemoglobin': '12.0',
            'Heart Rate': '86',
            'Height': '170.0',
            'Lactate': '1.9',
            'Lactate dehydrogenase': '280',
            'Lactic acid': '1.9',
            'Magnesium': '1.7',
            'Mean blood pressure': '77.0',
            'Mean corpuscular hemoglobin': '30',
            'Mean corpuscular hemoglobin concentration': '34',
            'Mean corpuscular volume': '88',
            'Oxygen saturation': '98.0',
            'Partial pressure of cabon dioxide': '40',
            'Partial pressure of oxygen': '88',
            'Partial thromboplastin time': '70',
            'Peak inspiratory pressure': '12',
            'Phosphate': '4',
            'Platelets': '300',
            'Positive end-expiratory pressure': '0.5',
            'Potassium': '4.4',
            'Prothrombin time': '13.5',
            'Respiratory rate': '19',
            'Sodium': '140',
            'Systolic blood pressure': '118.0',
            'Temperature': '36.6',
            'Troponin-I': '0.03',
            'Troponin-T': '0.19',
            'Urine output': '1500',
            'Weight': '81.0',
            'White blood cell count': '8.0',
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

    # Full 65-column original header used for index mapping
    _ORI_HEADER = [
        'Hours', 'Alanine aminotransferase', 'Albumin', 'Alkaline phosphate',
        'Anion gap', 'Asparate aminotransferase', 'Basophils', 'Bicarbonate',
        'Bilirubin', 'Blood culture', 'Blood urea nitrogen', 'Calcium',
        'Calcium ionized', 'Capillary refill rate', 'Chloride', 'Cholesterol',
        'Creatinine', 'Diastolic blood pressure', 'Eosinophils',
        'Fraction inspired oxygen', 'Glascow coma scale eye opening',
        'Glascow coma scale motor response', 'Glascow coma scale total',
        'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height',
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

    def _setup_bins(self, N_channels):
        """Compute begin_pos, end_pos, and cur_len for the channel layout."""
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
        return begin_pos, end_pos, cur_len

    def _write(self, data, bin_id, channel, value, begin_pos):
        """Write a value into the discretized data array using regex parsing."""
        channel_id = self._channel_to_id[channel]
        if (self._is_categorical_channel[channel]):
            category_id = self._possible_values[channel].index(value)
            N_values = len(self._possible_values[channel])
            one_hot = np.zeros((N_values,))
            one_hot[category_id] = 1
            for pos in range(N_values):
                data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos]
        else:
            num_filter = re.compile(r'\d{1,2}[\,\.]{1}\d{1,2}')
            num_box = num_filter.findall(value)
            if len(num_box) != 0:
                value = num_box[0]
            else:
                value = self._normal_values[channel]
            data[bin_id, begin_pos[channel_id]] = float(value)

    def _impute(self, data, mask, original_value, N_bins, begin_pos):
        """Apply the imputation strategy to fill missing bins."""
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
                    self._write(data, bin_id, channel, imputed_value, begin_pos)

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
                    self._write(data, bin_id, channel, imputed_value, begin_pos)

    def _build_new_header(self):
        """Build the output header string."""
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
        return new_header

    def transform(self, X, header=None, end=None):
        if (header == None):
            header = self._header
        assert header[0] == "Hours"
        indexbox = []
        oriheader = self._ORI_HEADER
        for x in header:
            if x in oriheader:
                indexbox.append(oriheader.index(x))
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1] + eps

        if (self._start_time == 'relative'):
            first_time = ts[0]
        elif (self._start_time == 'zero'):
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if (end == None):
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time
        N_bins = int(max_hours / self._timestep + 1.0 - eps)

        begin_pos, end_pos, cur_len = self._setup_bins(N_channels)

        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0

        for row in X:
            t = float(row[0]) - first_time
            if (t > max_hours + eps):
                continue
            bin_id = int(t / self._timestep - eps)
            assert(bin_id >= 0 and bin_id < N_bins)

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
                self._write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]

        # impute missing values
        self._impute(data, mask, original_value, N_bins, begin_pos)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])

        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        if (self._store_masks):
            data = np.hstack([data, mask.astype(np.float32)])

        new_header = self._build_new_header()
        return (data, new_header)

    def print_statistics(self):
        print ("statistics of discretizer:")
        print ("\tconverted %d examples" % self._done_count)
        print ("\taverage unused data = %.2f percent" % (100.0 * self._unused_data_sum / self._done_count))
        print ("\taverage empty  bins = %.2f percent" % (100.0 * self._empty_bins_sum / self._done_count))

    def transform_first_t_hours(self, X, header=None, end=None):
        if (header == None):
            header = self._header
        assert header[0] == "Hours"
        indexbox = []
        oriheader = self._ORI_HEADER
        for x in header:
            if x in oriheader:
                indexbox.append(oriheader.index(x))
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i + 1] + eps

        if (self._start_time == 'relative'):
            first_time = ts[0]
        elif (self._start_time == 'zero'):
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if (end == None):
            max_hours = max(ts) - first_time
        else:
            if (end > 48):
                end = 48
            max_hours = end - first_time
        N_bins = int(max_hours / self._timestep + 1.0 - eps)

        begin_pos, end_pos, cur_len = self._setup_bins(N_channels)

        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0

        for row in X:
            t = float(row[0]) - first_time
            if (t > max_hours + eps):
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
                self._write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]

        # impute missing values
        self._impute(data, mask, original_value, N_bins, begin_pos)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])

        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        if (self._store_masks):
            data = np.hstack([data, mask.astype(np.float32)])

        new_header = self._build_new_header()
        return (data, new_header)

    def transform_end_t_hours(self, X, header=None, los=None, max_length=48):
        if (header == None):
            header = self._header
        assert header[0] == "Hours"
        indexbox = []
        oriheader = self._ORI_HEADER
        for x in header:
            if x in oriheader:
                indexbox.append(oriheader.index(x))
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

        begin_pos, end_pos, cur_len = self._setup_bins(N_channels)

        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0

        for row in X:
            t = float(row[0]) - first_time
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
                self._write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]

        # impute missing values
        self._impute(data, mask, original_value, N_bins, begin_pos)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])

        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        if (self._store_masks):
            data = np.hstack([data, mask.astype(np.float32)])

        new_header = self._build_new_header()
        return (data, new_header)

    def transform_remove_mask(self, X, header=None, los=None, max_length=48):
        if (header == None):
            header = self._header
        assert header[0] == "Hours"
        indexbox = []
        oriheader = self._ORI_HEADER
        for x in header:
            if x in oriheader:
                indexbox.append(oriheader.index(x))
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

        begin_pos, end_pos, cur_len = self._setup_bins(N_channels)

        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0

        for row in X:
            t = float(row[0]) - first_time
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
                self._write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]

        # impute missing values
        self._impute(data, mask, original_value, N_bins, begin_pos)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])

        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        # NOTE: no mask appended (unlike other transforms)
        new_header = self._build_new_header()
        return (data, new_header)

    def transform_reg(self, X, header=None, end=None):
        if (header == None):
            header = self._header
        assert header[0] == "Hours"
        indexbox = []
        oriheader = self._ORI_HEADER
        for x in header:
            if x in oriheader:
                indexbox.append(oriheader.index(x))
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i + 1] + eps

        if (self._start_time == 'relative'):
            first_time = ts[0]
        elif (self._start_time == 'zero'):
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if (end == None):
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time
        N_bins = int(max_hours / self._timestep + 1.0 - eps)

        begin_pos, end_pos, cur_len = self._setup_bins(N_channels)

        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0

        for row in X:
            t = float(row[0]) - first_time
            if (t > max_hours + eps):
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
                self._write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]

        # impute missing values
        self._impute(data, mask, original_value, N_bins, begin_pos)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])

        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        new_header = self._build_new_header()
        return (data, new_header, begin_pos, end_pos)

    def transform_end_t_hours_reg(self, X, header=None, los=None, max_length=48):
        if (header == None):
            header = self._header
        assert header[0] == "Hours"
        indexbox = []
        oriheader = self._ORI_HEADER
        for x in header:
            if x in oriheader:
                indexbox.append(oriheader.index(x))
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

        begin_pos, end_pos, cur_len = self._setup_bins(N_channels)

        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0

        for row in X:
            t = float(row[0]) - first_time
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
                self._write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]

        # impute missing values
        self._impute(data, mask, original_value, N_bins, begin_pos)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])

        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        new_header = self._build_new_header()
        return data, mask.astype(np.float32)

    def missing_data(self, X, header=None, length=48):
        missing_d = 0
        stay_with_missing_d = 0

        if (header == None):
            header = self._header
        assert header[0] == "Hours"
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i + 1] + eps

        max_hours = length
        first_time = max(ts) - length
        if first_time < 0:
            missing_d += int(length - max(ts))
            stay_with_missing_d += 1

        return missing_d, stay_with_missing_d
