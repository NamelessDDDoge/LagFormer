import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import PositionalEncoding, LaggingMultiHeadAttention, PositionwiseFeedForward
import math

from torchdiffeq import odeint

class Model(nn.Module):
    def __init__(self, opt, time_num, d_model, d_inner, n_head, d_k, d_v,
                 alpha=0.3, use_lag_soft=False, lag_temp=1.0, dropout=0.1, nodes_num=None):
        super().__init__()
        # self.alpha = nn.Parameter(torch.tensor(alpha))
        self.lag_temp = lag_temp
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=1)
        self.position_enc = PositionalEncoding(d_hid=d_model, n_position=time_num)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.spa_attn = LaggingMultiHeadAttention(d_model, n_head, d_k, opt.lag, nodes_num)
        self.spa_attn = LaggingMultiHeadAttention(d_model, n_head, d_k, opt.lag, None)
        self.readout = nn.Linear(d_model, 1) 
        self.ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pred_len = opt.pred_len
        self.pred_forward = getattr(opt, 'pred_forward', 0)
        self.uProj = nn.Linear(d_model, d_model)
        self.uNorm = nn.LayerNorm(d_model, eps=1e-6)

    def generate_u(self, z):
        B, T, N, D = z.shape
        u = z.mean(dim=2).detach()
        u = self.uProj(u)
        u = self.uNorm(self.uProj(u)).reshape(-1, D).unsqueeze(1)
        return u
    

    def forward(self, x, method='rk4'):
        """
        x: [B, T, N]
        返回：
          y_hat: [B, T, N, total_pred]，对每个时间步的前后预测
          A: [N, N] 有效连接（常数）
        """

        # 1) 序列编码，用于生成注意力
        x_emb = self.conv1(x.unsqueeze(1)).permute(0, 2, 3, 1)          # [B, T, N, D]
        B, T, N, D = x_emb.shape
        pe = self.position_enc(x_emb).unsqueeze(2).expand(B, T, N, D)
        z0 = self.layer_norm(self.dropout(x_emb + pe))
        
        # 2) 得到动态系统矩阵 A
        adj_per_lag = self.spa_attn(z0)
        adj = adj_per_lag.mean(dim=0)
        N = adj.size(1)
        adj = adj * (1.0 - torch.eye(N, device=adj.device))
        
        u = self.generate_u(z0)
        def ode_func(t, z):
            z = z.view(-1, N, D)
            drift = torch.einsum('bnd,nm->bmd', z, adj)
            drift = drift + u
            return drift  

        z0_vec = z0.reshape(B * T, N, D)     
        
        # 处理预测
        if self.pred_forward > 0:
            # 1. 过去预测 (Backward): 时间反向积分 0 -> -1
            t_backward = torch.linspace(0, -1, self.pred_forward + 1, device=x.device)[1:]
            ode_backward = odeint(ode_func, z0_vec, t_backward, method=method) # [pred_forward, B*T, N, D]
            # 这里的顺序是 [t-1, t-2, ... t-pred_forward]，为了和数据集对齐，需要翻转
            ode_backward = torch.flip(ode_backward, dims=[0]) # 对齐为 [t-pred_forward, ... t-1]
            
            # 2. 未来预测 (Forward): 时间正向积分 0 -> 1
            t_forward = torch.linspace(0, 1, self.pred_len + 1, device=x.device)[1:]
            ode_forward = odeint(ode_func, z0_vec, t_forward, method=method) # [pred_len, B*T, N, D]
            
            # 3. 拼接
            odeint_out = torch.cat([ode_backward, ode_forward], dim=0) # [total_pred, B*T, N, D]
            total_pred = self.pred_forward + self.pred_len
        else:
            # 纯未来预测 (原逻辑)
            t_points = torch.linspace(0, 1, self.pred_len + 1, device=x.device)[1:]
            odeint_out = odeint(ode_func, z0_vec, t_points, method=method) # [pred_len, B * T, N, D]
            total_pred = self.pred_len

        preds_vec = odeint_out.reshape(total_pred, B, T, N, D).permute(1, 2, 3, 0, 4)   # [B, T, N, total_pred, D]
        
        y = self.ffn(preds_vec)
        y = self.readout(y).squeeze(-1)
        y = y.reshape(B, T, N, total_pred)
        
        return y, adj.T, adj_per_lag.swapaxes(1, 2)
