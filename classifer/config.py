import sys
sys.path.append('../')
from model import *

# global setting
RESULT_PATH = './result/'
SAVE_PATH = './tmp/'
SEED = 123456
EPOCH = 50
TRAIN = True
TEST = True
MODAL = ['word', 'visual', 'acoustic']
MODEL = {'lstm': LSTM,
         'eflstm': EFLSTM,
         'lflstm': LFLSTM,
         'tfn': TFN,
         'lmf': LMF,
         'mfn': MFN,
         'marn': MARN,
         'mmmu_ba': MMMU_BA}

# general model setting
model_name = 'mmmu_ba'
word_dim = 300
visual_dim = 47
acoustic_dim = 74
input_dims = {'word': word_dim, 'visual': visual_dim, 'acoustic': acoustic_dim}
hidden_dims = {'word': 300, 'visual': 300, 'acoustic': 300}
fc_dim = 100
output_dim = 1
dropout = 0.1
# marn
local_dims = {'word': 256, 'visual': 256, 'acoustic': 256}
att_num = 4
# lmf
rank = 4
# mmmu_ba
rnn_dropout = 0.1

# optimizer
lr = 0.001
weight_decay = 1e-4
step_size = 10
gamma = 0.1
grad_clip_value = 1.0