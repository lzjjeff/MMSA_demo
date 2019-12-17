import torch
import torch.nn as nn
import torch.nn.functional as F

CUDA = torch.cuda.is_available()


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