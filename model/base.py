import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

CUDA = torch.cuda.is_available()


class LSTM(nn.Module):
    """BiLSTM"""
    def __init__(self, input_dims, hidden_dims, output_dim, fc_dim, dropout, words_size=None, modal=None):
        super(LSTM, self).__init__()
        assert modal is not None

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.fc_dim = fc_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.modal = modal

        if 'word' in modal:
            self.wrnn = nn.LSTM(input_dims['word'], hidden_dims['word'], bidirectional=True, dropout=dropout)
        if 'visual' in modal:
            self.vrnn = nn.LSTM(input_dims['visual'], hidden_dims['visual'], bidirectional=True, dropout=dropout)
        if 'acoustic' in modal:
            self.arnn = nn.LSTM(input_dims['acoustic'], hidden_dims['acoustic'], bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(sum([hidden_dims[_modal] for _modal in modal])*2, output_dim)
        self.dp = nn.Dropout(dropout)

    def extract_features(self, sequence, lengths, rnn):
        packed_sequence = pack_padded_sequence(sequence, lengths)
        packed_h, (final_h, _) = rnn(packed_sequence)
        return final_h

    def fusion(self, sequences, lengths):
        batch_size = lengths.size(0)
        final_h = []
        if 'word' in self.modal:
            final_hw = self.extract_features(sequences['word'], lengths, self.wrnn)
            final_h.append(final_hw)
        if 'visual' in self.modal:
            final_hv = self.extract_features(sequences['visual'], lengths, self.vrnn)
            final_h.append(final_hv)
        if 'acoustic' in self.modal:
            final_ha = self.extract_features(sequences['acoustic'], lengths, self.arnn)
            final_h.append(final_ha)
        h = torch.cat(final_h, dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        return h

    def forward(self, sequences, lengths):
        h = self.fusion(sequences, lengths)
        h = self.dp(h)
        o = self.fc(h)
        return o


class EFLSTM(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dim, fc_dim, dropout, words_size=None, modal=None):
        super(EFLSTM, self).__init__()
        assert modal is not None

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.fc_dim = fc_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.modal = modal
        input_size = sum([input_dims[_modal] for _modal in modal])
        hidden_size = sum([hidden_dims[_modal] for _modal in modal])

        self.rnn = nn.LSTM(input_size, hidden_size, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.dp = nn.Dropout(dropout)

    def forward(self, sequences, lengths):
        sequence = torch.cat([sequences[modal] for modal in self.modal], dim=2)
        packed_sequence = pack_padded_sequence(sequence, lengths)
        packed_h, (final_h, _) = self.rnn(packed_sequence)
        h = self.dp(final_h)
        o = self.fc(h).view(-1, 1)
        return o


class LFLSTM(nn.Module):
    """LFLSTM"""
    def __init__(self, input_dims, hidden_dims, output_dim, fc_dim, dropout, words_size=None, modal=None):
        super(LFLSTM, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.fc_dim = fc_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.modal = modal

        assert modal is not None

        if words_size:
            self.emb = nn.Embedding(words_size, input_dims[0])
        if 'word' in modal:
            self.wrnn1 = nn.LSTM(input_dims['word'], hidden_dims['word'], bidirectional=True)
            self.wrnn2 = nn.LSTM(2*hidden_dims['word'], hidden_dims['word'], bidirectional=True)
        if 'visual' in modal:
            self.vrnn1 = nn.LSTM(input_dims['visual'], hidden_dims['visual'], bidirectional=True)
            self.vrnn2 = nn.LSTM(2*hidden_dims['visual'], hidden_dims['visual'], bidirectional=True)
        if 'acoustic' in modal:
            self.arnn1 = nn.LSTM(input_dims['acoustic'], hidden_dims['acoustic'], bidirectional=True)
            self.arnn2 = nn.LSTM(2*hidden_dims['acoustic'], hidden_dims['acoustic'], bidirectional=True)
        self.fc1 = nn.Linear(sum([hidden_dims[_modal] for _modal in modal])*4, fc_dim)
        self.fc2 = nn.Linear(fc_dim, output_dim)
        self.dp = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.wlayer_norm = nn.LayerNorm((hidden_dims['word']*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_dims['visual']*2,))
        self.alayer_norm = nn.LayerNorm((hidden_dims['acoustic']*2,))
        self.bn = nn.BatchNorm1d(sum([hidden_dims[_modal] for _modal in modal])*4)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths)
        packed_h1, (final_h1, _) = rnn1(packed_sequence)
        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)
        _, (final_h2, _) = rnn2(packed_normed_h1)
        return [final_h1, final_h2]

    def fusion(self, sequences, lengths):
        batch_size = lengths.size(0)
        final_h = []
        if 'word' in self.modal:
            final_hw = self.extract_features(sequences['word'], lengths, self.wrnn1, self.wrnn2, self.wlayer_norm)
            final_h += final_hw
        if 'visual' in self.modal:
            final_hv = self.extract_features(sequences['visual'], lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
            final_h += final_hv
        if 'acoustic' in self.modal:
            final_ha = self.extract_features(sequences['acoustic'], lengths, self.arnn1, self.arnn2, self.alayer_norm)
            final_h += final_ha
        h = torch.cat(final_h, dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        return self.bn(h)

    def forward(self, sequences, lengths):
        h = self.fusion(sequences, lengths)
        h = self.fc1(h)
        h = self.dp(h)
        h = self.relu(h)
        o = self.fc2(h)
        return o


class GME(nn.Module):
    def __init__(self):
        super(GME, self).__init__()
        pass

    def forward(self, *input):
        pass



