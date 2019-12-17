import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from functools import reduce

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

        self.wrnn = nn.LSTM(input_dims['word'], hidden_dims['word'], bidirectional=True)
        self.vrnn = nn.LSTM(input_dims['visual'], hidden_dims['visual'], bidirectional=True)
        self.arnn = nn.LSTM(input_dims['acoustic'], hidden_dims['acoustic'], bidirectional=True)
        self.fc = nn.Linear(sum([hidden_dims[_modal] for _modal in modal])*2, output_dim)
        self.dp = nn.Dropout(dropout)

    def extract_features(self, sequence, lengths, rnn):
        packed_sequence = pack_padded_sequence(sequence, lengths)
        packed_h, (final_h, _) = rnn(packed_sequence)
        return final_h

    def fusion(self, sequences, lengths):
        batch_size = lengths.size(0)
        final_hw = self.extract_features(sequences['word'], lengths, self.wrnn)
        final_hv = self.extract_features(sequences['visual'], lengths, self.vrnn)
        final_ha = self.extract_features(sequences['acoustic'], lengths, self.arnn)
        h = torch.cat([final_hw, final_hv, final_ha], dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        return h

    def forward(self, sequences, lengths):
        h = self.fusion(sequences, lengths)
        h = self.dp(h)
        o = self.fc(h)
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
        self.wrnn1 = nn.LSTM(input_dims['word'], hidden_dims['word'], bidirectional=True)
        self.wrnn2 = nn.LSTM(2 * hidden_dims['word'], hidden_dims['word'], bidirectional=True)
        self.vrnn1 = nn.LSTM(input_dims['visual'], hidden_dims['visual'], bidirectional=True)
        self.vrnn2 = nn.LSTM(2 * hidden_dims['visual'], hidden_dims['visual'], bidirectional=True)
        self.arnn1 = nn.LSTM(input_dims['acoustic'], hidden_dims['acoustic'], bidirectional=True)
        self.arnn2 = nn.LSTM(2 * hidden_dims['acoustic'], hidden_dims['acoustic'], bidirectional=True)
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
        final_hw = self.extract_features(sequences['word'], lengths, self.wrnn1, self.wrnn2, self.wlayer_norm)
        final_hv = self.extract_features(sequences['visual'], lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        final_ha = self.extract_features(sequences['acoustic'], lengths, self.arnn1, self.arnn2, self.alayer_norm)
        h = torch.cat([final_hw, final_hv, final_ha], dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        return self.bn(h)

    def forward(self, sequences, lengths):
        h = self.fusion(sequences, lengths)
        h = self.fc1(h)
        h = self.dp(h)
        h = self.relu(h)
        o = self.fc2(h)
        return o


class MFN(nn.Module):
    """Memory Fusion Network"""
    def __init__(self, input_dims, hidden_dims, output_dim, mem_dim, dropout, modal=None):
        super(MFN, self).__init__()
        assert modal is not None

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.mem_dim = mem_dim
        self.dropout = dropout
        self.modal = modal
        h_dim = sum([hidden_dims[_modal] for _modal in modal])

        self.rnncell = nn.ModuleDict()
        for _modal in self.modal:
            self.rnncell[_modal] = nn.LSTMCell(input_dims[_modal], hidden_dims[_modal])
        self.att_fc1 = nn.Linear(h_dim*2, h_dim)
        self.att_fc2 = nn.Linear(h_dim, h_dim*2)
        self.att_dropout = nn.Dropout(dropout)
        self.gamma1_fc1 = nn.Linear(h_dim*2, h_dim)
        self.gamma1_fc2 = nn.Linear(h_dim, mem_dim)
        self.gamma1_dropout = nn.Dropout(dropout)
        self.gamma2_fc1 = nn.Linear(h_dim*2, h_dim)
        self.gamma2_fc2 = nn.Linear(h_dim, mem_dim)
        self.gamma2_dropout = nn.Dropout(dropout)
        self.update_fc1 = nn.Linear(h_dim*2, h_dim)
        self.update_fc2 = nn.Linear(h_dim, mem_dim)
        self.update_dropout = nn.Dropout(dropout)
        self.output_fc1 = nn.Linear(h_dim+mem_dim, h_dim+mem_dim)
        self.output_fc2 = nn.Linear(h_dim+mem_dim, output_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, sequences, lengths):
        N = list(sequences.values())[0].size(1)     # batch size
        T = list(sequences.values())[0].size(0)     # time steps
        h_t = {modal: torch.zeros(N, self.hidden_dims[modal]) for modal in self.modal}
        c_t = {modal: torch.zeros(N, self.hidden_dims[modal]) for modal in self.modal}
        mem = torch.zeros(N, self.mem_dim)
        all_h_t = {modal: [] for modal in self.modal}
        all_c_t = {modal: [] for modal in self.modal}
        all_mems = []

        if CUDA:
            for modal in self.modal:
                h_t[modal] = h_t[modal].cuda()
                c_t[modal] = c_t[modal].cuda()
            mem = mem.cuda()

        for t in range(T):
            prev_h_t = []
            prev_c_t = []
            # lstm step
            for modal in self.modal:
                prev_c_t.append(c_t[modal])
                h_t[modal], c_t[modal] = self.rnncell[modal](sequences[modal][t], (h_t[modal], c_t[modal]))
                all_h_t[modal].append(h_t[modal])
                all_c_t[modal].append(c_t[modal])
            # delta-memory attention
            prev_cc_t = torch.cat(prev_c_t, dim=1)
            cc_t = torch.cat([all_c_t[modal][-1] for modal in self.modal], dim=1)   # cat of three modal
            cc_tt = torch.cat([prev_cc_t, cc_t], dim=1)
            attn = F.softmax(self.att_fc2(self.att_dropout(F.relu(self.att_fc1(cc_tt)))), dim=1)
            cc_star = cc_tt * attn
            both = torch.cat([cc_star, mem], dim=1)
            # multi-view gated memory
            gamma1 = F.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(cc_star)))))
            gamma2 = F.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(cc_star)))))
            u_star = F.tanh(self.update_fc2(self.update_dropout(F.relu(self.update_fc1(cc_star)))))
            u_t = gamma1 * mem + gamma2 * u_star
            # update memory
            mem = u_t
            all_mems.append(u_t)

        hs_T = [torch.cat([all_h_t[modal][lengths[i]-1][i].unsqueeze(0) for i in range(N)], dim=0) for modal in self.modal]
        h_T = torch.cat(hs_T, dim=1)
        mem = torch.cat([all_mems[lengths[i]-1][i].unsqueeze(0) for i in range(N)], dim=0)
        o = torch.cat([h_T, mem], dim=1)
        o = self.output_fc2(self.output_dropout(torch.relu(self.output_fc1(o))))
        return o


class MARN(nn.Module):
    """Multi-attention Recurrent Network"""
    def __init__(self, input_dims, hidden_dims, output_dim, local_dims, att_num, dropout, modal=None):
        super(MARN, self).__init__()
        assert modal is not None
        self.modal = modal
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.att_num = att_num
        self.total_h_dim = sum([hidden_dims[_modal] for _modal in modal])
        self.total_local_dim = sum([local_dims[_modal] for _modal in modal])

        self.rnncell = nn.ModuleDict()
        self.dim_reduce_model = nn.ModuleDict()
        for _modal in self.modal:
            self.rnncell[_modal] = LSTHM(hidden_dims[_modal], input_dims[_modal], self.total_h_dim)
            self.dim_reduce_model[_modal] = nn.Linear(self.att_num*self.hidden_dims[_modal], local_dims[_modal])
        self.att_model = nn.Linear(self.total_h_dim, self.att_num*self.total_h_dim)
        self.hybrid_fc = nn.Linear(self.total_local_dim, self.total_h_dim)
        self.output_fc = nn.Linear(self.total_h_dim, output_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, sequences, lengths):
        N = list(sequences.values())[0].size(1)  # batch size
        T = list(sequences.values())[0].size(0)  # time steps
        h_t = {modal: torch.zeros(N, self.hidden_dims[modal]) for modal in self.modal}
        c_t = {modal: torch.zeros(N, self.hidden_dims[modal]) for modal in self.modal}
        z_t = torch.zeros(N, self.total_h_dim)
        all_h_t = {modal: [] for modal in self.modal}
        all_c_t = {modal: [] for modal in self.modal}
        all_z_t = []
        if CUDA:
            for modal in self.modal:
                h_t[modal] = h_t[modal].cuda()
                c_t[modal] = c_t[modal].cuda()
            z_t = z_t.cuda()

        for t in range(T):
            # lsthm step
            for modal in self.modal:
                c_t[modal], h_t[modal] = self.rnncell[modal].step(sequences[modal][t], c_t[modal], h_t[modal], z_t)
                all_c_t[modal].append(c_t[modal])
                all_h_t[modal].append(h_t[modal])
            hc_t = torch.cat(list(h_t.values()), dim=1)
            # MAB step
            attns = F.softmax(self.att_model(hc_t).view(N, self.att_num, -1), dim=2)    # k attentions
            kh_t = hc_t.view(N, 1, -1).expand(N, self.att_num, self.total_h_dim) * attns     # k hidden vectors
            s_t = []
            for i in range(len(self.modal)):
                m_s = 0 if i is 0 else sum([self.hidden_dims[self.modal[j]] for j in range(i)]) # start index
                m_e = sum([self.hidden_dims[self.modal[j]] for j in range(i+1)])                # end index
                h_mt = kh_t[:, :, m_s:m_e].contiguous().view(N, -1)
                s_mt = self.dim_reduce_model[self.modal[i]](h_mt)
                s_t.append(s_mt)
            s_t = torch.cat(s_t, dim=1)
            z_t = self.hybrid_fc(s_t)
            all_z_t.append(z_t)
        hs_T = [torch.cat([all_h_t[modal][lengths[i] - 1][i].unsqueeze(0) for i in range(N)], dim=0) for modal in self.modal]
        h_T = torch.cat(hs_T, dim=1)
        z_T = torch.cat([all_z_t[lengths[i] - 1][i].unsqueeze(0) for i in range(N)], dim=0)
        o = torch.cat([h_T, z_T], dim=1)
        o = self.output_fc(self.output_dropout(z_T))
        return o


class RMFN(nn.Module):
    pass


class TFN(nn.Module):
    """Tendor Fusion Network"""
    def __init__(self, input_dims, hidden_dims, output_dim, fc_dim, dropout, modal=None):
        super(TFN, self).__init__()
        assert modal is not None
        self.modal = modal
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.w_subnet = WordSubnet(input_dims['word'], hidden_dims['word'], dropout)
        self.v_subnet = Subnet(input_dims['visual'], hidden_dims['visual'], dropout)
        self.a_subnet = Subnet(input_dims['acoustic'], hidden_dims['acoustic'], dropout)
        self.fc1 = nn.Linear(reduce(lambda x, y: x*y, [dim+1 for dim in list(hidden_dims.values())]), fc_dim)
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

        z_w = self.w_subnet(sequences['word'], lengths)
        z_v = self.v_subnet(torch.mean(sequences['visual'], dim=0))
        z_a = self.a_subnet(torch.mean(sequences['acoustic'], dim=0))
        _z_w = torch.cat([ones, z_w], dim=1)
        _z_v = torch.cat([ones, z_v], dim=1)
        _z_a = torch.cat([ones, z_a], dim=1)
        all_z = [_z_w, _z_v, _z_a]

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
            self.modality_factor[_modal] = nn.Parameter(torch.Tensor(rank, hidden_dims[_modal]+1, fc_dim))
            if CUDA:
                self.modality_factor[_modal] = self.modality_factor[_modal].cuda()
        self.fusion_weights = nn.Parameter(torch.Tensor(1, rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, fc_dim))
        self.fc = nn.Linear(fc_dim, output_dim)
        self.dp = nn.Dropout(dropout)

        # init factors
        for _modal in modal:
            nn.init.xavier_normal(self.modality_factor[_modal])
        nn.init.xavier_normal(self.fusion_weights)
        nn.init.zeros_(self.fusion_bias)

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
        o = o.view(-1, self.fc_dim)
        o = self.dp(o)
        o = self.fc(o)
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


class GME(nn.Module):
    def __init__(self):
        super(GME, self).__init__()
        pass

    def forward(self, *input):
        pass


class LSTHM(nn.Module):

    def __init__(self, cell_size, in_size, hybrid_in_size):
        super(LSTHM, self).__init__()
        self.cell_size = cell_size
        self.in_size = in_size
        self.W = nn.Linear(in_size, 4 * self.cell_size)
        self.U = nn.Linear(cell_size, 4 * self.cell_size)
        self.V = nn.Linear(hybrid_in_size, 4 * self.cell_size)

    def step(self, x, ctm1, htm1, ztm1):
        input_affine = self.W(x)
        output_affine = self.U(htm1)
        hybrid_affine = self.V(ztm1)

        sums = input_affine + output_affine + hybrid_affine

        # biases are already part of W and U and V
        f_t = F.sigmoid(sums[:, :self.cell_size])
        i_t = F.sigmoid(sums[:, self.cell_size:2 * self.cell_size])
        o_t = F.sigmoid(sums[:, 2 * self.cell_size:3 * self.cell_size])
        ch_t = F.tanh(sums[:, 3 * self.cell_size:])
        c_t = f_t * ctm1 + i_t * ch_t
        h_t = F.tanh(c_t) * o_t
        return c_t, h_t
