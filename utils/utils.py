from scipy.stats import pearsonr
import scipy.io as sio
import torch
import numpy as np
import os
import random
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Read all txt file data in the directory, each txt file represents a subject

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def get_checkpoint_dir(opt, timestamp, name='Baseline', outer=False):
    folder = f'{name}_{timestamp}_idx{opt.index}_dmodel{opt.d_model}_nhead{opt.n_head}_lag{opt.lag}_predlen{opt.pred_len}_alpha{opt.alpha}_beta{opt.beta}_gamma{opt.gamma}_inner{opt.d_inner_hid}_epoch{opt.epoch}_batchsize{opt.batch_size}_dropout{opt.dropout}'
    
    checkpoint_dir = os.path.join("./checkpoints", folder)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def get_adj_per_lag(model, data_loader, device):
    """获取模型的滞后注意力矩阵"""
    model.eval()
    with torch.no_grad():
        data_tmp, _ = next(iter(data_loader))
        data_tmp = data_tmp.to(device)
        X = data_tmp[:, :-1]  # [B, T-1, N]
        
        x_emb = model.conv1(X.unsqueeze(1)).permute(0, 2, 3, 1)
        pe = model.position_enc(x_emb).unsqueeze(2).expand_as(x_emb)
        X_spa = model.layer_norm(model.dropout(x_emb + pe))
        
        adj_per_lag = model.spa_attn(X_spa, None, squeeze=False)  # [L, N, N]
        return adj_per_lag.cpu().numpy()


def load_data_txt(path):
    '''return: data [M, T, N], M: number of samples'''
    data = []
    all_path = os.listdir(path)
    m = len(all_path)
    for sub_path in all_path:
        position = path + '\\' + sub_path
        file = open(position, 'r')
        file_data = file.readlines()
        file_data = file_data[1:]
        t = len(file_data)
        for row in file_data:
            tmp_list = row.split('\t')
            tmp_list[-1] = tmp_list[-1].replace('\n','')
            tmp_list = np.array([float(x) for x in tmp_list])
            n = len(tmp_list)
            data.append(tmp_list)

    data = np.array(data).reshape((m, t, n))

    return data

def load_sim(path):
    data_mat = sio.loadmat(path)
    N = data_mat['Nnodes'][0, 0] # N is a 1×1 ndarray
    B = data_mat['Nsubjects'][0, 0]
    T = data_mat['Ntimepoints'][0, 0]
    data = data_mat['ts'].reshape((B, T, N))
    ground_truth = data_mat['net']
    ground_truth = ground_truth[0, :, :]

    index = (ground_truth == -1)
    ground_truth[index] = 0
    index = (ground_truth > 0)
    ground_truth[index] = 1

    return data, ground_truth # ground_truth is a 0-1 matrix


# data:[M, T, N]
def cal_pearson(data):
    m, t, n = data.shape
    adj = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            for k in range(m):
                pear, _ = pearsonr(data[k, :, i], data[k, :, j])
                adj[i, j] += pear
            adj[i, j] = adj[i, j]/m
            adj[j, i] = adj[i, j]
    return adj



def sliding_window_cutting(data, window_size, overlap):
    step = window_size - overlap
    B, T, N = data.shape

    nums = (T - window_size)//step + 1
    if (T - window_size) % step != 0:
        nums += 1
    new_B = B * nums

    data_sliced = torch.FloatTensor(new_B, window_size, N)
    for b in range(B):
        for num in range(nums - 1):
            data_sliced[b*nums + num, :, :] = data[b, step*num:step*num+window_size, :]
        data_sliced[b * nums + nums - 1, :, :] = data[b, -window_size:, :]

    return data_sliced





def change01(adj, threshold):
    alpha = copy.deepcopy(adj)
    N = alpha.shape[0]
    alpha = np.where(alpha >= threshold, 1, 0)
    for i in range(N):
        alpha[i, i] = 0
    return alpha

def cal_metrics(pre, ground_truth):    
    ground_truth = (ground_truth == 1) # change to bool matrix
    pre = (pre == 1)
    TP = np.sum(np.sum(pre & ground_truth))
    FP = np.sum(np.sum(pre & (~ground_truth)))
    FN = np.sum(np.sum((~pre) & ground_truth))
    TN = np.sum(np.sum((~pre) & (~ground_truth)))
    # pre_tmp = np.transpose(pre)
    # RA = np.sum(np.sum(pre_tmp & ground_truth))
    # print("TP, FP, FN, TN:", TP, FP, FN, TN)
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    SHD = FP + FN
    return precision, recall, F1, accuracy, SHD

def softThres(adj, soft_threshold):
    N = adj.shape[0]
    cur = adj + np.diag([10000000] * N)
    min_val = np.min(cur)
    max_val = np.max(adj)
    thresh = min_val + soft_threshold * (max_val - min_val)
    return thresh
    
