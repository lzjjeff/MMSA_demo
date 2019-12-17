import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

CUDA = torch.cuda.is_available()


class RMFN(nn.Module):
    #TODO