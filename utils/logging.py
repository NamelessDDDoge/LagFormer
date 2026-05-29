import logging
import os
from datetime import datetime
import numpy as np


def setup_logger(opt, timestamp=None, name='Baseline', folder=None):
    """设置日志记录器"""
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    folder = f'{name}_{timestamp}_idx{opt.index}_dmodel{opt.d_model}_nhead{opt.n_head}_lag{opt.lag}_predlen{opt.pred_len}_predforward{opt.pred_forward}_alpha{opt.alpha}_beta{opt.beta}_gamma{opt.gamma}_inner{opt.d_inner_hid}_epoch{opt.epoch}_batchsize{opt.batch_size}_dropout{opt.dropout}'
        
    log_filename = f"./logs/{folder}/train.log"
    print(log_filename)
    # 创建logs目录（如果不存在）
    os.makedirs(f"logs/{folder}", exist_ok=True)
    
    # 创建一个唯一的logger名称，避免重复使用同一个logger
    logger_name = f"logger_{name}_{opt.index}_{timestamp}"
    logger = logging.getLogger(logger_name)
    
    # 清除现有的handlers，确保不会重复添加
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 设置logger级别
    logger.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建文件handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 添加handlers到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 防止日志传播到根logger（避免重复输出）
    logger.propagate = False

    
    return logger, log_filename

def log_metrics_summary(logger, metrics):
    """计算并记录metrics的均值和标准差，返回mu和std"""
    mu = np.mean(metrics, axis=0)
    std = np.std(metrics, axis=0)
    names = ['precision', 'recall', 'F1', 'accuracy', 'SHD']
    
    for i in range(len(names)):
        logger.info(f'{names[i]}: {mu[i]}')
    logger.info(f'std:{std}')
    
    return mu, std, names


def get_main_path(log_path):
    """从log_path中提取main_path"""
    prefix, suffix = log_path.split('idx')
    suffix = suffix[2:]
    main_path = prefix + suffix
    os.makedirs(main_path, exist_ok=True)
    return main_path


def write_index_results(main_path, idx, mu, names):
    """写入单个index的结果"""
    with open(f'{main_path}/result.txt', 'a') as f:
        f.write(f'--------------index {idx}--------------\n')
        for i in range(len(names)):
            f.write(f'{names[i]}: {mu[i]:.3f}\n')


def write_summary_results(main_path, mu_all, std_all, names):
    """写入汇总结果"""
    with open(f'{main_path}/result.txt', 'a') as f:
        f.write(f'--------------Summary--------------\n')
        for i in range(len(names)):
            f.write(f'{names[i]}: {mu_all[i]:.3f}\n')
        f.write(f'std:{std_all}')
