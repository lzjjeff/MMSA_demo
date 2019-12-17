import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from functools import reduce

CUDA = torch.cuda.is_available()


class TFN(nn.Module):
    """Tendor Fusion Network"""
    def __init__(self, input_dims, hidden_dims, output_dim, fc_dim, dropout, modal=None):
        super(TFN, self).__init__()
        assert modal is not None
        self.modal = modal
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.subnet = nn.ModuleDict()
        for _modal in modal:
            if _modal is 'word':
                self.subnet[_modal] = WordSubnet(input_dims[_modal], hidden_dims[_modal], dropout)
            else:
                self.subnet[_modal] = Subnet(input_dims[_modal], hidden_dims[_modal], dropout)
        self.fc1 = nn.Linear(reduce(lambda x, y: x*y, [dim+1 for dim in [hidden_dims[_modal] for _modal in modal]]), fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, output_dim)
        self.dp = nn.Dropout(dropout)

    def fusion(self, all_z):
        batch_size = all_z[0].size(0)
        product = all_z[0]
        for z in all_z[1:]:
            product = torch.matmul(product.unsqueeze(dim=2), z.unsqueeze(dim=1))
            product = product.view(batch_size, -1)
        return product

    def forward(self, sequences, lengths):
        N = list(sequences.values())[0].size(1)  # batch size
        if CUDA:
            ones = torch.ones(N, requires_grad=False).unsqueeze(dim=1).cuda()
        else:
            ones = torch.ones(N, requires_grad=False).unsqueeze(dim=1)

        all_z = []
        for modal in self.modal:
            if modal is 'word':
                z = self.subnet[modal](sequences[modal], lengths)
            else:
                z = self.subnet[modal](torch.mean(sequences[modal], dim=0))
            _z = torch.cat([ones, z], dim=1)
            all_z.append(_z)

        fusioned_z = self.fusion(all_z)
        dropped = self.dp(fusioned_z)
        o = F.relu(self.fc1(dropped))
        o = F.relu(self.fc2(o))
        o = self.fc3(o)
        return o


class LMF(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dim, fc_dim, rank, dropout, modal=None):
        super(LMF, self).__init__()
        assert modal is not None
        self.modal = modal
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.fc_dim = fc_dim
        self.rank = rank

        self.subnet = nn.ModuleDict()
        self.modality_factor = {}
        for _modal in modal:
            if _modal is 'word':
                self.subnet[_modal] = WordSubnet(input_dims[_modal], hidden_dims[_modal], dropout)
            else:
                self.subnet[_modal] = Subnet(input_dims[_modal], hidden_dims[_modal], dropout)
            self.modality_factor[_modal] = nn.Parameter(torch.Tensor(rank, hidden_dims[_modal]+1, output_dim))
            if CUDA:
                self.modality_factor[_modal] = self.modality_factor[_modal].cuda()
        self.fusion_weights = nn.Parameter(torch.Tensor(1, rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, output_dim))

        # init factors
        for _modal in modal:
            nn.init.xavier_normal(self.modality_factor[_modal])
        nn.init.uniform_(self.fusion_weights)
        nn.init.uniform_(self.fusion_bias)

    def fusion(self, all_z):
        fusioned = []
        for modal in self.modal:
            fusioned.append(torch.matmul(all_z[modal], self.modality_factor[modal]))
        fusioned_z = reduce(lambda x,y: x*y, fusioned)
        return fusioned_z

    def forward(self, sequences, lengths):
        N = list(sequences.values())[0].size(1)  # batch size
        if CUDA:
            ones = torch.ones(N, requires_grad=False).unsqueeze(dim=1).cuda()
        else:
            ones = torch.ones(N, requires_grad=False).unsqueeze(dim=1)

        all_z = {}
        for modal in self.modal:
            if modal is 'word':
                all_z[modal] = self.subnet[modal](sequences[modal], lengths)
            else:
                all_z[modal] = self.subnet[modal](torch.mean(sequences[modal], dim=0))
            all_z[modal] = torch.cat([ones, all_z[modal]], dim=1)

        fusioned_z = self.fusion(all_z)
        o = torch.matmul(self.fusion_weights, fusioned_z.permute(1, 0, 2)).squeeze() + self.fusion_bias
        o = o.view(-1, self.output_dim)
        return o


class Subnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(Subnet, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dp = nn.Dropout(dropout)

    def forward(self, input):
        normed = self.bn(input)
        droped = self.dp(normed)
        h = F.relu(self.fc1(droped))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return h


class WordSubnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(WordSubnet, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=1, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dp = nn.Dropout(dropout)

    def forward(self, sequence, lengths):
        packed_sequence = pack_padded_sequence(sequence, lengths)
        packed_h, (final_h, _) = self.rnn(packed_sequence)
        h = self.dp(final_h.squeeze())
        h = self.fc(h)
        return h