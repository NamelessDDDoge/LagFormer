import torch
import torch.nn as nn
import torch.nn.functional as F

class loss_func(nn.Module):
    def __init__(self, alpha_sp):
        super(loss_func, self).__init__()
        self.alpha_sp = alpha_sp
        # self.alpha_acy = alpha_acy

    def forward(self, output, adj, label):
        L_sp = torch.sum(torch.sum(torch.abs(adj)))
        criterion = nn.MSELoss()
        L_pre = criterion(output, label)
        return L_pre + self.alpha_sp * L_sp


def spatial_contrastive_loss(z_pred, z_target, temperature=0.1):
    """
    z_pred: [B, T, N, D] - 预测的隐状态
    z_target: [B, T, N, D] - 真实未来的隐状态 (Target Embeddings)
    """
    B, T, N, D = z_pred.shape
    
    # 展平 B 和 T，因为我们只关心同一时刻 N 个节点之间的区分度
    # [M, N, D], where M = B*T
    z_pred = z_pred.view(-1, N, D)    
    z_target = z_target.view(-1, N, D)
    
    # 1. 归一化 (Cosine Similarity 准备)
    z_pred = F.normalize(z_pred, dim=-1)
    z_target = F.normalize(z_target, dim=-1)
    
    # 2. 计算相似度矩阵
    # 我们要计算 z_pred 中每个节点 i 与 z_target 中所有节点 j 的相似度
    # logits shape: [M, N, N] 
    # logits[m, i, j] = sim(pred_node_i, target_node_j)
    logits = torch.bmm(z_pred, z_target.transpose(1, 2))
    
    # print(np.round(logits[0].detach().cpu().numpy(), 2))
    
    # 除以温度系数
    logits /= temperature
    
    # 3. 构造标签 (Labels)
    # 正样本在对角线上：pred_node_i 应该和 target_node_i 最像
    # labels: [0, 1, 2, ..., N-1] for each sample in M
    labels = torch.arange(N, device=z_pred.device).unsqueeze(0).expand(B*T, -1) # [M, N]
    
    # 4. 计算 Cross Entropy Loss
    # PyTorch 的 CrossEntropyLoss 期望 input 是 [Batch, Classes]
    # 这里我们需要 reshape
    # Input (logits): [M * N, N] (把 N 个节点的预测任务展开)
    # Target (labels): [M * N]
    
    loss = F.cross_entropy(logits.reshape(-1, N), labels.reshape(-1))
    
    return loss



def structure_loss(W):
    """
    W: Adjacency matrix (num_nodes, num_nodes)
    该 Loss 旨在消除被间接路径解释的直接边。
    """
    
    # 2. 计算长度为 2 的路径强度
    # W_path2[i, j] = sum(|W_ik| * |W_kj|)
    if W.dim() == 2:
        W_path2 = torch.mm(W, W)
    elif W.dim() == 3:
        W_path2 = torch.bmm(W, W)
    else:
        raise ValueError("W must be a 2D or 3D tensor")
        
    penalty_matrix = W * W_path2 
    loss = torch.sum(penalty_matrix)
    
    return loss


def structure_super_loss(W):
    """
    W: Adjacency matrix (num_nodes, num_nodes) or (Batch, N, N)
    Loss = sum( W * (e^W - I - W) )
    旨在惩罚那些“已经可以通过间接路径解释”的直接连接。
    """
    
    # 1. 计算矩阵指数 E = e^W
    # torch.matrix_exp 支持 batch 操作，会自动处理 2D 或 3D 输入
    E = torch.matrix_exp(W)
    
    # 2. 构建单位矩阵 I
    N = W.size(-1)
    if W.dim() == 2:
        I = torch.eye(N, device=W.device)
    else: # 3D case: (B, N, N)
        I = torch.eye(N, device=W.device).unsqueeze(0)
        
    # 3. 计算所有间接路径 (长度 >= 2 的部分)
    # Indirect = (I + W + W^2/2! + ...) - I - W = W^2/2! + W^3/3! + ...
    indirect_paths = E - I - W
    
    # 4. 计算惩罚项
    # 只有当 直连边(W) 和 间接路径(indirect_paths) 同时存在时，惩罚才大
    penalty_matrix = W * indirect_paths
    
    loss = torch.sum(penalty_matrix)
    
    return loss

class MSEsparseLoss(nn.Module):
    def __init__(self, alpha, beta=0, gamma=0):
        super(MSEsparseLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, output, adj, adj_per_lag, label, x, x_pred, eps=1e-8):
        N = adj.shape[0]
        L_sp = torch.sum(torch.log(torch.abs(adj) / eps + 1)) / (N - 1)
        # L_sp = torch.sum(torch.abs(adj * (1 - torch.eye(N, device=adj.device))))
        mse = nn.MSELoss()
        
        L_pre = mse(output, label)
        L_rec = 0 if x_pred is None else mse(x, x_pred)
            
        L_stru = structure_loss(adj_per_lag)
           
        return L_pre + self.alpha * L_sp + self.beta * L_stru + self.gamma * L_rec


class CrossLoss(nn.Module):
    def __init__(self, alpha_sp, beta_cross):
        super(CrossLoss, self).__init__()
        self.alpha_sp = alpha_sp
        self.beta_cross = beta_cross

    def forward(self, output, adj, Z, E, label, eps=1e-8):
        N = adj.shape[0]
        L_sp = torch.sum(torch.log(torch.abs(adj) / eps + 1)) / (N - 1)
        
        Zc = Z - Z.mean(0, keepdim=True)
        Ec = E - E.mean(0, keepdim=True)
        Zc = Zc / (Zc.std(0, keepdim=True) + eps)
        Ec = Ec / (Ec.std(0, keepdim=True) + eps)
        C = (Zc.T @ Ec) / (Z.size(0) - 1)           # [Dz, De]
        L_2 = (C**2).mean()
        
        mse = nn.MSELoss()
        L_pre = mse(output, label)
        # print(self.beta_cross * L_2)
        res = L_pre + self.alpha_sp * L_sp + self.beta_cross * L_2
        return res

def get_loss_function(opt):
    """
    Factory function to get the loss function based on options.

    支持两种损失函数：
    1. MSEsparseLoss: 原始损失函数
    2. RegularizedLoss: 新增传递路径惩罚和入度均衡惩罚

    使用方法：
    - 原始: alpha=0.8, beta=0.0, gamma=0.0, delta=0.0
    - Phase1实验: alpha=0.8, beta=0.0, gamma=0.1, delta=0.1
    """
    # 如果gamma或delta非零，使用新的RegularizedLoss
    gamma = getattr(opt, 'gamma', 0.0)
    delta = getattr(opt, 'delta', 0.0)

    if gamma != 0.0 or delta != 0.0:
        path_method = getattr(opt, 'path_method', 'matrix_exp')
        degree_mode = getattr(opt, 'degree_mode', 'combined')
        return RegularizedLoss(
            alpha=opt.alpha,
            beta=opt.beta,
            gamma=gamma,
            delta=delta,
            path_method=path_method,
            degree_mode=degree_mode
        )
    else:
        return MSEsparseLoss(alpha=opt.alpha, beta=opt.beta, gamma=opt.gamma)


# ============================================================================
# Phase 1: 正则化损失函数 - 区分直接因果 vs 间接因果
# ============================================================================

def path_loss(W, method='matrix_exp'):
    """
    高阶传递路径惩罚 - 区分直接因果 vs 间接因果
    核心思想：直接因果 A 不满足传递性 A² ≠ A⊙A

    W: Adjacency matrix, shape [N, N] or [B, N, N]
    返回: 惩罚标量

    可微分实现：
    - method='hadamard': L = ||W @ W - W ⊙ W||²  (W² 对角线元素来自直接边*直接边)
    - method='matrix_exp': L = sum(W * (exp(W) - I - W))  (更精确的传递路径估计)
    """
    if W.dim() == 2:
        W_sq = torch.mm(W, W)  # 矩阵乘法
        W_had = W * W  # Hadamard乘积
    else:
        W_sq = torch.bmm(W, W)
        W_had = W * W

    if method == 'hadamard':
        # 差分形式：如果存在传递路径但无直接边，惩罚
        diff = W_sq - W_had
        loss = torch.sum(diff ** 2)
    elif method == 'matrix_exp':
        # 矩阵指数形式: exp(W) = I + W + W²/2! + W³/3! + ...
        N = W.size(-1)
        if W.dim() == 2:
            I = torch.eye(N, device=W.device)
        else:
            I = torch.eye(N, device=W.device).unsqueeze(0)

        E = torch.matrix_exp(W)
        indirect = E - I - W  # 所有长度>=2的路径

        # 只有当直连边和间接路径同时存在时才惩罚
        penalty = W * indirect
        loss = torch.sum(penalty)
    else:
        raise ValueError(f"Unknown method: {method}")

    return loss


def degree_loss(W, target_degree=None, mode='combined', penalty_strength=0.1):
    """
    入度/出度均衡惩罚 - 避免"单原因捷径"

    W: Adjacency matrix, shape [N, N]
    target_degree: 目标入度（可选）
    mode:
      - 'balance': 惩罚入度方差，鼓励均衡
      - 'min_degree': 惩罚入度为0的情况
      - 'max_degree': 惩罚入度过大（避免单原因）
      - 'combined': 组合 (balance + min_degree)

    返回: 惩罚标量（可微）
    """
    # 入度 = 按列求和 (j->i 的边，i是行，j是列)
    in_degree = W.sum(dim=0)  # [N]
    # 出度 = 按行求和
    out_degree = W.sum(dim=1)  # [N]

    if mode == 'balance':
        # 惩罚入度方差，鼓励均衡分布
        mean_degree = in_degree.mean()
        variance = ((in_degree - mean_degree) ** 2).mean()
        # 也考虑出度均衡
        out_mean = out_degree.mean()
        out_var = ((out_degree - out_mean) ** 2).mean()
        loss = variance + out_var

    elif mode == 'min_degree':
        # 惩罚入度为0（鼓励每个节点至少有一个原因）
        # 使用 softplus 确保可微
        loss = -penalty_strength * torch.sum(torch.log(in_degree + 1e-8))

    elif mode == 'max_degree':
        # 惩罚入度过大（避免单原因）
        max_degree = in_degree.max()
        if target_degree is not None:
            loss = torch.relu(max_degree - target_degree)
        else:
            loss = max_degree

    elif mode == 'combined':
        # 组合：均衡 + 最小入度
        mean_degree = in_degree.mean()
        variance = ((in_degree - mean_degree) ** 2).mean()
        zero_penalty = torch.relu(1.0 - in_degree).sum()  # 入度<1时惩罚
        loss = variance + penalty_strength * zero_penalty

    else:
        loss = torch.tensor(0.0, device=W.device)

    return loss


class RegularizedLoss(nn.Module):
    """
    组合正则化损失函数 - Phase 1 核心

    L_total = L_pred
            + α * L_sparse(邻接矩阵稀疏性)
            + β * L_structure(结构损失，矩阵指数)
            + γ * L_path(传递路径惩罚，区分直接vs间接)
            + δ * L_degree(入度均衡，避免单原因捷径)
    """

    def __init__(self, alpha=0.1, beta=0.0, gamma=0.0, delta=0.0, path_method='matrix_exp', degree_mode='combined'):
        super().__init__()
        self.alpha = alpha    # 稀疏性
        self.beta = beta      # 结构（已有）
        self.gamma = gamma    # 传递路径惩罚
        self.delta = delta    # 入度均衡
        self.path_method = path_method
        self.degree_mode = degree_mode

    def forward(self, output, adj, adj_per_lag, label, x, x_pred, eps=1e-8):
        """
        adj: [N, N] 平均邻接矩阵
        adj_per_lag: [lag, N, N] 按lag的邻接矩阵
        """
        N = adj.shape[0]

        # 1. 基础预测损失
        mse = nn.MSELoss()
        L_pre = mse(output, label)

        # 2. 稀疏性损失（已有）
        L_sp = torch.sum(torch.log(torch.abs(adj) / eps + 1)) / (N - 1)

        # 3. 结构损失（已有）
        L_stru = structure_loss(adj_per_lag)

        # 4. 传递路径惩罚（新增）
        L_path = path_loss(adj, method=self.path_method)

        # 5. 入度均衡惩罚（新增）
        L_degree = degree_loss(adj, mode=self.degree_mode)

        # 组合
        total = (L_pre
                 + self.alpha * L_sp
                 + self.beta * L_stru
                 + self.gamma * L_path
                 + self.delta * L_degree)

        return total
