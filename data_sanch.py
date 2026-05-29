import os
import torch
import numpy as np


def get_sanch(index):
    path = f'./dataset/sanch/Network{index}_amp/data_fslfilter'
    ground_truth = np.loadtxt(f'./dataset/sanch/Network{index}_amp/ground_truth.txt', delimiter='\t')
    all_path = os.listdir(path)
    subjects = len(all_path)
    data = np.empty((subjects, 0, 0))
    
    for i, sub_path in enumerate(all_path):
        position = path + '/' + sub_path
        data_tmp = np.loadtxt(position, skiprows=1, delimiter='\t')
        if i == 0:
            data = np.expand_dims(data_tmp, axis=0)
        else:
            data = np.concatenate((data, np.expand_dims(data_tmp, axis=0)), axis=0)

    # data:[S, T, N]
    data = torch.FloatTensor(data)
    # data = (data - data.mean()) / data.std()
    return data, ground_truth
    
    
def prepare_sanch(opt):
    data, ground_truth = get_sanch(opt.index)
    device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    pred_len = opt.pred_len
    pred_forward = getattr(opt, 'pred_forward', 0)
    
    # [S, T, N ] → [S, N, T]
    data = data.permute(0, 2, 1)
    
    # 支持双向预测，x 是中心，y 包含过去和未来
    window_size = pred_forward + 1 + pred_len
    data_windows = data.unfold(dimension=2, size=window_size, step=1)        # [S, N, T_out, window_size]
    
    # x 取中心点
    x = data_windows[:, :, :, pred_forward]                                 # [S, N, T_out]
    
    if pred_forward > 0:
        y_past = data_windows[:, :, :, :pred_forward]      # y_backward: 过去的数据
        y_future = data_windows[:, :, :, pred_forward+1:] # y_forward: 未来数据
        y = torch.cat([y_past, y_future], dim=-1)         # [S, N, T_out, pred_forward + pred_len]
    else:
        y = data_windows[:, :, :, 1:]                     # 纯后向预测 (原逻辑)
    
    x = x.permute(0, 2, 1)            # [S, T, N]
    y = y.permute(0, 2, 1, 3)         # [S, T, N, total_pred]
    opt.time_num = x.shape[1]
    
    dataset = torch.utils.data.TensorDataset(x, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    return data_loader, ground_truth
    
    
def prepare_sanch_aug(opt):
    data, ground_truth = get_sanch(opt.index)
    device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    pred_len = opt.pred_len
    pred_forward = getattr(opt, 'pred_forward', 0)
    
    # [S, T, N ] → [S, N, T]
    data = data.permute(0, 2, 1)
    window_size = pred_forward + 1 + pred_len
    data_windows = data.unfold(dimension=2, size=window_size, step=1)
    
    x = data_windows[:, :, :, pred_forward]
    if pred_forward > 0:
        y_past = data_windows[:, :, :, :pred_forward]
        y_future = data_windows[:, :, :, pred_forward+1:]
        y = torch.cat([y_past, y_future], dim=-1)
    else:
        y = data_windows[:, :, :, 1:]
        
    x = x.permute(0, 2, 1)            # [S, T, N]
    y = y.permute(0, 2, 1, 3)         # [S, T, N, total_pred]
    opt.time_num = x.shape[1]
    
    dataset = torch.utils.data.TensorDataset(x, y)
