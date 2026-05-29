import argparse
import numpy as np
import random
import os
import copy
import gc

import torch
from torch import nn
import torch.optim as optim

import warnings
from datetime import datetime

from model import Model
# from model.NeuralDCM import NeuralDCM

from optim import ScheduledOptim
from data_sanch import prepare_sanch

from utils.utils import *
from utils.losses import get_loss_function
from utils.logging import setup_logger, log_metrics_summary, get_main_path, write_index_results, write_summary_results
from utils.plotter import plot_metrics_for_run, plot_adj_per_lag_heatmaps, plot_prediction_comparison


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings('ignore')


def train_epoch(model, data_loader, optimizer, criterion, epoch_idx):
    model.train()
    train_loss = []
    batch_adj = []
    batch_adj_per_lag = []
    
    for data_tmp, label_tmp in data_loader:
        X = data_tmp
        Y = label_tmp
        
        optimizer.zero_grad()
        output, adj, adj_per_lag = model(X)
        # X_pred = model.reconstruct(X)
        X_pred = None
            
        loss = criterion(output, adj, adj_per_lag, Y, X, X_pred)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step_and_update_lr()

        train_loss.append(loss.item())
        batch_adj.append(adj.cpu().detach().numpy())
        batch_adj_per_lag.append(adj_per_lag.cpu().detach().numpy())
            
    train_loss = np.average(train_loss)
    adj_mean = np.mean(batch_adj, axis=0)
    adj_per_lag = np.mean(batch_adj_per_lag, axis=0)

    return train_loss, adj_mean, adj_per_lag


def train(data_loader, ground_truth, device, opt, logger, checkpoint_dir, run_id=None):
    model = Model(
        opt=opt,
        time_num=opt.time_num,
        d_model=opt.d_model,
        d_inner=opt.d_inner_hid,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        nodes_num=opt.num_nodes
    ).to(device)
    
    
    optimizer = ScheduledOptim(
        optim.Adam([{'params': model.parameters()}], betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps
    )
    criterion = get_loss_function(opt).to(device)

    if run_id is not None:
        model_save_dir = os.path.join(checkpoint_dir, f'run_{run_id}', 'models')
    else:
        model_save_dir = os.path.join(checkpoint_dir, 'models')
    os.makedirs(model_save_dir, exist_ok=True)

    # 准备固定的可视化数据 (取前2个样本)
    viz_batch_size = 2
    viz_dataset = data_loader.dataset
    real_viz_size = min(len(viz_dataset), viz_batch_size)
    viz_X, viz_Y = viz_dataset[:real_viz_size] 
    viz_X = viz_X.to(device)
    viz_Y = viz_Y.to(device)

    # 训练循环
    for epoch_i in range(opt.epoch):
        train_loss, adj, adj_per_lag = train_epoch(model, data_loader, optimizer, criterion, epoch_i)
        # 处理邻接矩阵
        N = adj.shape[0]
        adj[np.arange(N), np.arange(N)] = 0
        adj_per_lag[:, np.arange(N), np.arange(N)] = 0
        opt.threshold = softThres(adj, opt.soft_threshold)
        adj_binary = change01(adj, threshold=opt.threshold)
        
        # 计算指标
        precision, recall, F1, accuracy, SHD = cal_metrics(adj_binary, ground_truth)
        
        # 定期日志输出
        if (epoch_i + 1) % 10 == 0:    
            logger.info(f'index:{opt.index}, epoch:{epoch_i+1}, loss:{train_loss: .3f}, '
                       f'precision:{precision:.6f}, recall:{recall:.6f}, F1:{F1:.6f}, '
                       f'accuracy:{accuracy:.6f}, SHD:{SHD:.2f}')
            logger.info(f'train loss:{train_loss}')
            logger.info(f"threshold:{opt.threshold}")
            logger.info('------------Predicted-Raw:------------')
            logger.info(f'\n{np.round(adj, 4)}')
            logger.info('------------Predicted:------------')
            logger.info(f'\n{adj_binary}')
            logger.info('----------Ground Truth:-----------')
            logger.info(f'\n{ground_truth.astype(int)}')
        
        gc.collect()
        torch.cuda.empty_cache() 
        
    # 保存模型并可视化
    model_save_path = os.path.join(model_save_dir, f'model_epoch_{epoch_i+1}.pt')
    torch.save({
        'epoch': epoch_i + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        'metrics': {
            'precision': precision,
            'recall': recall,
            'F1': F1,
            'accuracy': accuracy,
            'SHD': SHD
        }
    }, model_save_path)
    
    # 绘制adj_per_lag热力图
    plot_adj_per_lag_heatmaps(adj_per_lag, epoch_i+1, checkpoint_dir, opt.threshold, ground_truth, run_id)
    
    # 绘制预测结果
    model.eval()
    with torch.no_grad():
        viz_output, _, _ = model(viz_X)
        T = viz_output.shape[1]
        pred_len = viz_output.shape[-1]
        indices = range(0, T, pred_len)
        viz_preds_selected = viz_output[:, indices, :, :].cpu().numpy()
        viz_targets_selected = viz_Y[:, indices, :, :].cpu().numpy()
        plot_prediction_comparison(viz_preds_selected, viz_targets_selected, epoch_i+1, checkpoint_dir, run_id)
    model.train() 
        
    return adj, precision, recall, F1, accuracy, SHD


def run_experiments(data_loader, ground_truth, device, opt, logger, log_path, checkpoint_dir, runs):
    """对单个数据集运行多次实验，返回metrics列表"""
    metrics = []
    
    for i in range(1, runs + 1):
        logger.info(f'************************************runs:{i}************************************')
        adj, precision, recall, F1, accuracy, SHD = train(
            data_loader, 
            ground_truth, 
            device, 
            opt,
            logger,
            checkpoint_dir,
            run_id=i
        )
        metrics.append([precision, recall, F1, accuracy, SHD])
        plot_metrics_for_run(log_path, i)
    
    return metrics


def run_all_indices(device, opt, runs):
    """运行所有index (1-4) 的实验"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mu_all = np.zeros(5)
    std_all = np.zeros(5)
    names = None
    log_path = None
    
    for idx in range(4, 0, -1):
        opt.index = str(idx)
        
        # 初始化logger和数据
        logger, log_path = setup_logger(opt, timestamp, name=opt.model_name)
        checkpoint_dir = get_checkpoint_dir(opt, timestamp, name=opt.model_name)
        
        data_loader, ground_truth = prepare_sanch(opt)
            
        opt.num_nodes = data_loader.dataset[0][0].shape[-1]
        
        # 运行实验
        metrics = run_experiments(
            data_loader, ground_truth, device, opt, 
            logger, log_path, checkpoint_dir, runs
        )
        
        # 统计并记录
        mu, std, names = log_metrics_summary(logger, metrics)
        mu_all += mu
        std_all += std
        
        # 写入单个index结果
        main_path = get_main_path(log_path)
        write_index_results(main_path, idx, mu, names)
    
    # 写入汇总结果
    mu_all /= 4
    std_all /= 4
    main_path = get_main_path(log_path)
    write_summary_results(main_path, mu_all, std_all, names)


def run_single_index(device, opt, runs):
    """运行单个index的实验"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 初始化logger和数据
    logger, log_path = setup_logger(opt, None, name=opt.model_name)
    checkpoint_dir = get_checkpoint_dir(opt, timestamp, name=opt.model_name)
    
    # 根据是否开启数据增强选择数据准备函数
    data_loader, ground_truth = prepare_sanch(opt)
        
    opt.num_nodes = data_loader.dataset[0][0].shape[-1]
    
    # 运行实验
    metrics = run_experiments(
        data_loader, ground_truth, device, opt, 
        logger, log_path, checkpoint_dir, runs
    )
    
    # 统计并记录
    log_metrics_summary(logger, metrics)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    
    # 数据相关参数
    parser.add_argument('-skiprows', type=int, default=1, help='in np.loadtxt, the num of skiprows')
    parser.add_argument('--pred_len', type=int, default=5, help='后向预测长度')
    parser.add_argument('--pred_forward', type=int, default=0, help='前向预测长度')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--index', type=str, default='1', help='the index of fMRI dataset')
    parser.add_argument('-time_num', type=int, default=None)
    parser.add_argument("--nodes_num", type=int, default=None)
    
    # 训练相关参数
    parser.add_argument('--epoch', type=int, default=301)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)
    parser.add_argument('--lr_mul', type=float, default=1.2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('-label_smoothing', action='store_true')
    
    # 损失函数参数
    parser.add_argument('-soft_threshold', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.8)   # 稀疏性
    parser.add_argument('--beta', type=float, default=0.0)      # 结构损失
    parser.add_argument('--gamma', type=float, default=0.0)    # 传递路径惩罚 (区分直接vs间接)
    parser.add_argument('--delta', type=float, default=0.0)     # 入度均衡惩罚 (避免单原因捷径)
    parser.add_argument('--path_method', type=str, default='matrix_exp', help='传递路径惩罚方法: hadamard 或 matrix_exp')
    parser.add_argument('--degree_mode', type=str, default='combined', help='degree mode: balance, min_degree, max_degree, combined')
    
    # 模型结构参数
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--d_inner_hid', type=int, default=64)
    parser.add_argument('--lag', type=str, default='10', help='Length of the input window (int or comma-separated list like 0,1,4,8)')
    parser.add_argument("--model_name", default="Baseline", type=str)
    parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of filter-enhanced blocks")
    parser.add_argument("--num_gru_layers", default=2, type=int, help="number of gru layers")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)
    parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--initializer_range", default=0.02, type=float)
    parser.add_argument("--no_filters", action="store_true",
                        help="if no filters, filter layers transform to self-attention")
    
    # 优化器参数
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam second beta value")
    
    # 其他参数
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--variance", default=5, type=float)
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")

    opt = parser.parse_args()
    
    # 解析 lag 参数
    if ',' in opt.lag:
        opt.lag = [int(x.strip()) for x in opt.lag.split(',')]
    else:
        opt.lag = int(opt.lag)
    
    # 设置随机种子
    set_seed(opt.seed)
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    # 设置设备
    device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # 配置模型参数
    assert opt.d_model % opt.n_head == 0, "d_model must be divisible by n_head"
    opt.d_k = opt.d_model // opt.n_head
    opt.d_v = opt.d_model // opt.n_head
    opt.soft_threshold = 0.5
    
    # 运行实验
    runs = 20
    if opt.index == '0':
        run_all_indices(device, opt, runs)
    else:
        run_single_index(device, opt, runs)


if __name__ == '__main__':
    main()
