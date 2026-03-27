g_map = {'F': 1, 'M': 2}

e_map = {
    'ASIAN': 1,
    'BLACK': 2,
    'HISPANIC': 3,
    'WHITE': 4,
    'OTHER': 5,
    'UNABLE TO OBTAIN': 0,
    'PATIENT DECLINED TO ANSWER': 0,
    'UNKNOWN': 0,
    '': 0,
}

i_map = {
    'Government': 0,
    'Self Pay': 1,
    'Medicare': 2,
    'Private': 3,
    'Medicaid': 4,
}

AGE_MEANS = 93.0
AGE_STD = 23.99553529900332
CONT_CHANNELS = [2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]

# Hyperparameter defaults
DEFAULT_HPARAMS = {
    'l1': 0,
    'l2': 0,
    'dim': 16,
    'depth': 2,
    'epochs': 50,
    'dropout': 0.3,
    'batch_size': 20,
    'timestep': 1.0,
    'target_repl': 0.0,
    'rec_dropout': 0.0,
    'target_repl_coef': 0.0,
    'task': 'ihm',
    'network': 'lstm_cnn',
    'loss': 'binary_crossentropy',
    'loss_weights': None,
}
