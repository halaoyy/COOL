import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(torch.nn.Module):
    def __init__(self, hid_dim, sim_func):
        super(Attention, self).__init__()
        self.sim_func = sim_func
        if self.sim_func == 'dot':
            self.w = nn.Linear(hid_dim, hid_dim)
            # pass
        elif self.sim_func == 'dot-linear':
            self.w = nn.Linear(hid_dim, hid_dim)
            self.v = nn.Linear(hid_dim, hid_dim)
        elif self.sim_func == 'concat':
            self.w = nn.Linear(hid_dim * 2, hid_dim)
            self.v = nn.Linear(hid_dim, 1)
        elif self.sim_func == 'add':
            self.w = nn.Linear(hid_dim, hid_dim)
            self.u = nn.Linear(hid_dim, hid_dim)
            self.v = nn.Linear(hid_dim, 1)
        elif self.sim_func == 'linear':
            self.w = nn.Linear(hid_dim, hid_dim)
        self._init_weights()

    def forward(self, query, seq, seq_lens=None, sign="pos"):
        if self.sim_func == 'dot':
            # query = self.w(query)
            query = query.squeeze().unsqueeze(1)
            if sign == "pos":
                a = self.mask_softmax(torch.bmm(seq, query.permute(0, 2, 1)), seq_lens, 1)
            else:
                a = -self.mask_softmax(-torch.bmm(seq, query.permute(0, 2, 1)), seq_lens, 1)
            return a
        elif self.sim_func == 'dot-linear':
            query = self.w(query)
            query = query.squeeze().unsqueeze(1)
            seq = self.v(seq)
            if sign == "pos":
                a = self.mask_softmax(torch.bmm(seq, query.permute(0, 2, 1)), seq_lens, 1)
            else:
                a = -self.mask_softmax(-torch.bmm(seq, query.permute(0, 2, 1)), seq_lens, 1)
            return a
        elif self.sim_func == 'concat':
            seq_len = len(seq[0])
            batch_size = len(seq)
            query = query.squeeze().unsqueeze(1)
            a = torch.cat([seq, query.repeat([1, seq_len, 1])], 2).reshape([seq_len * batch_size, -1])
            a = F.relu(self.w(a))
            a = F.relu(self.v(a))
            a = self.mask_softmax(a.reshape([batch_size, seq_len, 1]), seq_lens, 1)
            return a
        elif self.sim_func == 'add':
            seq_len = len(seq[0])
            batch_size = len(seq)
            seq = self.w(seq.reshape([batch_size * seq_len, -1]))
            query = self.u(query).repeat([seq_len, 1])
            a = self.mask_softmax(self.v(F.tanh(seq + query)).reshape([batch_size, seq_len, 1]), seq_lens, 1)
            return a
        elif self.sim_func == 'linear':
            seq_len = len(seq[0])
            batch_size = len(seq)
            query = query.squeeze()
            query = self.w(query).unsqueeze(2)
            a = self.mask_softmax(torch.bmm(seq, query), seq_lens, 1)
            return a

    def _init_weights(self):
        if self.sim_func == 'dot':
            pass
        elif self.sim_func == 'dot-linear':
            nn.init.uniform_(self.w.weight, a=-0.1, b=0.1)
            nn.init.uniform_(self.v.weight, a=-0.1, b=0.1)
        elif self.sim_func == 'concat':
            nn.init.uniform_(self.w.weight, a=-0.1, b=0.1)
            nn.init.uniform_(self.v.weight, a=-0.1, b=0.1)
        elif self.sim_func == 'add':
            nn.init.uniform_(self.w.weight, a=-0.1, b=0.1)
            nn.init.uniform_(self.u.weight, a=-0.1, b=0.1)
            nn.init.uniform_(self.v.weight, a=-0.1, b=0.1)
        elif self.sim_func == 'linear':
            nn.init.uniform_(self.w.weight, a=-0.1, b=0.1)

    def mask_softmax(self, seqs, seq_lens=None, dim=1):
        if seq_lens is None:
            res = F.softmax(seqs, dim=dim)
        else:
            max_len = len(seqs[0])
            batch_size = len(seqs)
            ones = seq_lens.new_ones(batch_size, max_len, device=seq_lens.device)
            range_tensor = ones.cumsum(dim=1)
            mask = (seq_lens.unsqueeze(1) >= range_tensor).long()
            mask = mask.float()
            mask = mask.unsqueeze(2)
            # masked_vector = seqs.masked_fill((1 - mask).byte(), -1e32)
            masked_vector = seqs.masked_fill((1 - mask).bool(), -1e32)
            res = F.softmax(masked_vector, dim=dim)
        return res
