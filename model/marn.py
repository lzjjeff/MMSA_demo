import torch
import torch.nn as nn
import torch.nn.functional as F

CUDA = torch.cuda.is_available()


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