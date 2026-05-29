import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x.size(1)].clone().detach() # [1, n_position, d_hid]


class LaggingMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_k, lag, nodes_num=None):
        super(LaggingMultiHeadAttention, self).__init__()
        self.lag = lag
        self.n_head = n_head
        self.d_k = d_k
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.temperature = d_k ** 0.5
        
        self.lagFuse = nn.Parameter(torch.zeros(lag+1))
        self.A = nn.Parameter(torch.ones(nodes_num, nodes_num)) if nodes_num is not None else None
        
    def forward(self, x):
        # x: [B, T, N, d_model]
        q = x.unfold(dimension=1, size=self.lag+1, step=1)                      # 包含同期的因果, lag=0的情况
        q = q.permute(0, 4, 1, 2, 3)                                            # [B, L, T-L, N, d_model]
        B, _, t, N, _ = q.shape
        x = x[:, self.lag:].unsqueeze(1).repeat(1, self.lag+1, 1, 1, 1)         # [B, L, T-L, N, d_model]
        
        # [B, L, T-L, h, N, d_k]
        q = F.gelu(self.w_qs(q).view(B, self.lag+1, t, N, self.n_head, self.d_k).transpose(3, 4))
        k = F.gelu(self.w_ks(x).view(B, self.lag+1, t, N, self.n_head, self.d_k).transpose(3, 4))
        
        # [B, L, T-L, h, N, N]
        logit = torch.matmul(q / self.temperature, k.transpose(-1, -2))
        
        attn = F.softmax(logit, dim=-1)
        # attn = F.tanh(logit)
        adj_per_lag = torch.mean(attn, dim=(0, 2, 3))          # [L, N, N]
        
        # score = F.softmax(self.lagFuse, dim=0)
        score = F.sigmoid(self.lagFuse)
        adj_per_lag = adj_per_lag * score[:, None, None]
        
        if self.A is not None:
            adj_per_lag = self.A.unsqueeze(0)
        
        # print(score)
        return adj_per_lag
    
    
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.leaky_relu(self.w_1(x), negative_slope=0.01))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        return x