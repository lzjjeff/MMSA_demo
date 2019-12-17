import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

CUDA = torch.cuda.is_available()


class MMMU_BA(nn.Module):
    """
        Multi-modal Multi-utterance - Bi-modal Attention
        2 or 3 modalities available
    """
    def __init__(self, input_dims, hidden_dims, output_dim, fc_dim, rnn_dropout, fc_dropout, modal=None):
        super(MMMU_BA, self).__init__()
        assert modal is not None
        self.modal = modal
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.rnn_dropout = rnn_dropout
        self.fc_dropout = fc_dropout

        self.rnn = nn.ModuleDict()
        self.fc = nn.ModuleDict()
        for _modal in self.modal:
            self.rnn[_modal] = nn.GRU(input_dims[_modal], hidden_dims[_modal], dropout=rnn_dropout, bidirectional=True)
            self.fc[_modal] = nn.Linear(2*hidden_dims[_modal], fc_dim)
        self.clf = nn.Linear(9*fc_dim, output_dim)

    def extract_features(self, sequence, lengths, rnn, fc):
        # packed_sequence = pack_padded_sequence(sequence, lengths)
        hs, final_h = rnn(sequence)
        # h = final_h.permute(1, 0, 2).contiguous().view(batch_size, -1)
        m = F.dropout(F.relu(fc(hs.squeeze())), self.fc_dropout)   # b, d
        return m

    def attenion(self, m1, m2):
        M1 = torch.matmul(m1, m2.permute(1, 0))
        M2 = torch.matmul(m2, m1.permute(1, 0))
        N1 = F.softmax(M1, dim=1)
        N2 = F.softmax(M2, dim=1)
        O1 = torch.matmul(N1, m2)
        O2 = torch.matmul(N2, m1)
        A1 = O1*m1
        A2 = O2*m2
        mm = torch.cat([A1, A2], dim=1)
        return mm

    def mean(self, sequence, lengths):
        sum = torch.sum(sequence, dim=0)    # b*d
        mean = sum / lengths.unsqueeze(1).type(torch.FloatTensor).cuda()
        return mean.unsqueeze(dim=1)

    def forward(self, sequences, lengths):
        N = list(sequences.values())[0].size(1)  # batch size
        all_m = {}  # A, V, T
        for modal in self.modal:
            # sequences[modal] = torch.sum(sequences[modal], dim=0).unsqueeze(dim=1)
            sequences[modal] = self.mean(sequences[modal], lengths)
            all_m[modal] = self.extract_features(sequences[modal], lengths, self.rnn[modal], self.fc[modal])

        keys = []   # source-target pair
        all_mm = []     # set of MMMU-BA_key
        if len(self.modal) == 2:
            all_mm.append(self.attenion(all_m[self.modal[0]], all_m[self.modal[1]]))
        elif len(self.modal) == 3:
            for modal1, modal2 in zip(self.modal, self.modal[1:]+self.modal[:1]):
                keys.append(modal1[0]+modal2[0])
                all_mm.append(self.attenion(all_m[modal1], all_m[modal2]))
        else:
            print("modality number error")
            exit(1)
        concat = torch.cat(all_mm+[all_m[modal] for modal in self.modal], dim=1)

        o = self.clf(concat)
        return o


class Utt_MMMU_BA(nn.Module):
    """
        Multi-modal Multi-utterance - Bi-modal Attention
        2 or 3 modalities available
        for Utterance-level MOSI dataset
    """
    def __init__(self, input_dims, hidden_dims, output_dim, fc_dim, rnn_dropout, fc_dropout, modal=None):
        super(Utt_MMMU_BA, self).__init__()
        assert modal is not None
        self.modal = modal
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.rnn_dropout = rnn_dropout
        self.fc_dropout = fc_dropout

        self.rnn = nn.ModuleDict()
        self.fc = nn.ModuleDict()
        for _modal in self.modal:
            self.rnn[_modal] = nn.GRU(input_dims[_modal], hidden_dims[_modal], dropout=rnn_dropout, bidirectional=True)
            self.fc[_modal] = nn.Linear(2*hidden_dims[_modal], fc_dim)
        self.clf = nn.Linear(9*fc_dim, output_dim)

    def extract_features(self, sequence, lengths, rnn, fc):
        packed_sequence = pack_padded_sequence(sequence, lengths)
        packed_h, final_h = rnn(packed_sequence)
        padded_h, _ = pad_packed_sequence(packed_h)
        h = padded_h.permute(1, 0, 2)
        m = F.dropout(F.relu(fc(h)), self.fc_dropout)   # b, d
        return m

    def attenion(self, m1, m2):
        M1 = torch.matmul(m1, m2.permute(0, 2, 1))
        M2 = torch.matmul(m2, m1.permute(0, 2, 1))
        N1 = F.softmax(M1, dim=2)
        N2 = F.softmax(M2, dim=2)
        O1 = torch.matmul(N1, m2)
        O2 = torch.matmul(N2, m1)
        A1 = O1*m1
        A2 = O2*m2
        mm = torch.cat([A1, A2], dim=2)
        return mm

    def forward(self, sequences, lengths):
        all_m = {}  # A, V, T
        for modal in self.modal:
            # sequences[modal] = torch.mean(sequences[modal], dim=0).unsqueeze(dim=1)
            all_m[modal] = self.extract_features(sequences[modal], lengths, self.rnn[modal], self.fc[modal])

        keys = []   # source-target pair
        all_mm = []     # set of MMMU-BA_key
        if len(self.modal) == 2:
            all_mm.append(self.attenion(all_m[self.modal[0]], all_m[self.modal[1]]))
        elif len(self.modal) == 3:
            for modal1, modal2 in zip(self.modal, self.modal[1:]+self.modal[:1]):
                keys.append(modal1[0]+modal2[0])
                all_mm.append(self.attenion(all_m[modal1], all_m[modal2]))
        else:
            print("modality number error")
            exit(1)
        concat = torch.cat(all_mm+[all_m[modal] for modal in self.modal], dim=2)

        o = self.clf(concat).view(concat.size(0)*concat.size(1), -1)

        # fetch non-pad
        index = torch.cat([torch.LongTensor(list(range(lengths[i]))) + i*concat.size(1) for i in range(len(lengths))],
                          dim=0).cuda()
        o = o.index_select(dim=0, index=index)
        return o
