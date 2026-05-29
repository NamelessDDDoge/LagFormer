import re
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from .utils import softThres

def plot_adj_per_lag_heatmaps(adj_per_lag, epoch, checkpoint_dir, threshold, ground_truth, run_id=None):
    """绘制滞后注意力矩阵的热力图（合并版本）
    
    第一排：各个时滞的热力图
    第二排：累计平均热力图
    
    Args:
        adj_per_lag: 注意力矩阵 [L, N, N]
        epoch: 当前训练轮数
        checkpoint_dir: 保存目录
        threshold: 当前epoch的softThres阈值
        ground_truth: 真实的邻接矩阵 [N, N]
        run_id: 运行ID
    """
    L, N, _ = adj_per_lag.shape
    
    # 创建保存热力图的目录
    if run_id is not None:
        heatmap_dir = os.path.join(checkpoint_dir, f'run_{run_id}', 'heatmaps')
    else:
        heatmap_dir = os.path.join(checkpoint_dir, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # 定义边框颜色和含义
    border_colors = {
        'green': 'TP',
        'red': 'FP',
        'blue': 'FN'
    }
    
    # 创建子图网格：2行，L列
    fig, axes = plt.subplots(2, L, figsize=(5*L, 10))
    fig.suptitle(f'Attention Matrices (Epoch {epoch}), Threshold: {threshold:.3f}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 如果L=1，需要确保axes是2维数组
    if L == 1:
        axes = axes.reshape(-1, 1)
    
    # 第一排：绘制每个lag的热力图
    for lag in range(L):
        # 对角线元素置0
        adj_lag = adj_per_lag[lag].copy()
        np.fill_diagonal(adj_lag, 0)
        
        # 计算最大绝对值用于设置vmin和vmax
        max_abs = max(abs(adj_lag.max() - threshold), 
                     abs(adj_lag.min() - threshold))
        vmin = threshold - max_abs
        vmax = threshold + max_abs
        
        # 绘制基础热力图
        ax = axes[0, lag]
        sns.heatmap(adj_lag, 
                    cmap='coolwarm',
                    center=threshold,
                    vmin=vmin,
                    vmax=vmax,
                    square=True,
                    xticklabels=range(1, N+1),
                    yticklabels=range(1, N+1),
                    ax=ax,
                    cbar_kws={'label': 'Attention'})
        
        # 添加不同颜色的边框
        for i in range(N):
            for j in range(N):
                if i != j:  # 跳过对角线
                    is_above_threshold = adj_lag[i, j] >= threshold
                    is_true_edge = ground_truth[i, j] == 1
                    
                    # 根据条件选择边框颜色
                    if is_above_threshold and is_true_edge:
                        color = 'green'  # TP
                    elif is_above_threshold and not is_true_edge:
                        color = 'red'    # FP
                    elif not is_above_threshold and is_true_edge:
                        color = 'blue'   # FN
                    else:
                        continue  # TN，不加边框
                    
                    # 添加边框
                    rect = plt.Rectangle([j, i], 1, 1,
                                      fill=False, edgecolor=color, linewidth=1.5)
                    ax.add_patch(rect)
        
        ax.set_title(f'Lag {lag}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Node', fontsize=11)
        ax.set_ylabel('Source Node', fontsize=11)
    
    # 第二排：绘制累计平均热力图
    for k in range(L):
        # 计算前(k+1)个lag的平均注意力矩阵
        avg_adj = np.mean(adj_per_lag[:k+1], axis=0).copy()
        np.fill_diagonal(avg_adj, 0)
        
        # 计算最大绝对值用于设置vmin和vmax
        max_abs = max(abs(avg_adj.max() - threshold), 
                     abs(avg_adj.min() - threshold))
        vmin = threshold - max_abs
        vmax = threshold + max_abs
        
        # 绘制热力图
        ax = axes[1, k]
        sns.heatmap(avg_adj, 
                   cmap='coolwarm',
                   center=threshold,
                   vmin=vmin,
                   vmax=vmax,
                   square=True,
                   xticklabels=range(1, N+1),
                   yticklabels=range(1, N+1),
                   ax=ax,
                   cbar_kws={'label': 'Attention'})
        
        # 添加不同颜色的边框
        for i in range(N):
            for j in range(N):
                if i != j:  # 跳过对角线
                    is_above_threshold = avg_adj[i, j] >= threshold
                    is_true_edge = ground_truth[i, j] == 1
                    
                    # 根据条件选择边框颜色
                    if is_above_threshold and is_true_edge:
                        color = 'green'  # TP
                    elif is_above_threshold and not is_true_edge:
                        color = 'red'    # FP
                    elif not is_above_threshold and is_true_edge:
                        color = 'blue'   # FN
                    else:
                        continue  # TN，不加边框
                    
                    # 添加边框
                    rect = plt.Rectangle([j, i], 1, 1,
                                      fill=False, edgecolor=color, linewidth=1.5)
                    ax.add_patch(rect)
        
        ax.set_title(f'Cumulative Avg (Lag 0-{k})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Node', fontsize=11)
        ax.set_ylabel('Source Node', fontsize=11)
    
    # 添加图例说明（在整个图的右侧）
    legend_elements = [plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=color, 
                                   linewidth=2, label=desc) 
                      for color, desc in border_colors.items()]
    fig.legend(handles=legend_elements, 
               loc='center left',
               bbox_to_anchor=(1.0, 0.5),
               title='Notations', 
               fontsize=12, 
               title_fontsize=13,
               frameon=True,
               fancybox=True,
               shadow=True)
    
    plt.tight_layout(rect=[0, 0, 0.98, 0.96])  # 留出空间给图例和总标题
    plt.savefig(os.path.join(heatmap_dir, f'epoch_{epoch}_all_lags.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_comparison(preds, targets, epoch, checkpoint_dir, run_id=None, samples_to_plot=None):
    """
    绘制预测结果和实际值的比较图
    
    Args:
        preds: 预测值 [B, Num_Segments, N, pred_len]
        targets: 真实值 [B, Num_Segments, N, pred_len]
        epoch: 当前训练轮数
        checkpoint_dir: 保存目录
        run_id: 运行ID
        samples_to_plot: 要绘制的样本索引列表，默认为前3个
    """
    if samples_to_plot is None:
        samples_to_plot = [0, 1, 2]
        
    B, Num_Segments, N, pred_len = preds.shape
    
    # 创建保存目录
    if run_id is not None:
        save_dir = os.path.join(checkpoint_dir, f'run_{run_id}', 'predictions')
    else:
        save_dir = os.path.join(checkpoint_dir, 'predictions')
    os.makedirs(save_dir, exist_ok=True)
    
    for b in samples_to_plot:
        if b >= B:
            continue
            
        # 准备数据
        # targets 需要拼接成完整的时间序列用于展示
        # [Num_Segments, N, pred_len] -> [Num_Segments, pred_len, N] -> [Num_Segments*pred_len, N]
        t_full = targets[b].transpose(0, 2, 1).reshape(-1, N)
        
        # preds 保持分段，以便用不同颜色绘制
        p_segments = preds[b] # [Num_Segments, N, pred_len]
        
        # 计算整张图的高度：每个节点一个小图 + 2个总图
        fig_height = (N + 2) * 2.5
        
        fig, axes = plt.subplots(N + 2, 1, figsize=(15, fig_height), sharex=True)
        if N + 2 == 1:
            axes = [axes]
            
        fig.suptitle(f'Time Series Prediction Analysis - Sample {b} (Epoch {epoch})', 
                    fontsize=16, fontweight='bold', y=1.002)
        
        # 1. 分别绘制每个节点的对比图
        for n in range(N):
            ax = axes[n]
            
            # 绘制真实值（连续线条）
            ax.plot(t_full[:, n], label='Ground Truth', color='black', linewidth=1.5, alpha=0.6)
            
            # 绘制预测值（分段线条）
            # 使用两种颜色交替
            colors = ['#F18F01', '#2E86AB'] # 橙色和深蓝色
            
            for i in range(Num_Segments):
                segment = p_segments[i, n, :] # [pred_len]
                x_range = range(i * pred_len, (i + 1) * pred_len)
                
                ax.plot(x_range, segment, color=colors[i % 2], linewidth=2, alpha=0.9)
                
                # 填充差异区域
                # 对应的真实值片段
                t_segment = targets[b, i, n, :]
                ax.fill_between(x_range, t_segment, segment, color=colors[i % 2], alpha=0.2)
            
            # 计算该节点的MSE
            p_full = preds[b, :, n, :].flatten()
            mse = np.mean((t_full[:, n] - p_full)**2)
            
            ax.set_ylabel(f'Node {n}', fontsize=12, fontweight='bold')
            ax.set_title(f'Node {n} Dynamics (MSE: {mse:.4f})', loc='left', fontsize=11)
            
            if n == 0:
                # 手动创建图例
                from matplotlib.lines import Line2D
                custom_lines = [Line2D([0], [0], color='black', lw=1.5, alpha=0.6),
                                Line2D([0], [0], color=colors[0], lw=2),
                                Line2D([0], [0], color=colors[1], lw=2)]
                ax.legend(custom_lines, ['Ground Truth', 'Pred Segment A', 'Pred Segment B'], 
                         loc='upper right', frameon=True, fancybox=True, shadow=True)
            
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        # 2. 绘制所有节点的预测值总图
        ax_pred = axes[N]
        node_colors = plt.cm.viridis(np.linspace(0, 1, N))
        
        # 为了清晰，这里只画连续的预测值（拼接后），虽然失去了分段信息但能看整体趋势
        p_full_all = preds[b].transpose(0, 2, 1).reshape(-1, N) # [Total_Len, N]
        
        for n in range(N):
            ax_pred.plot(p_full_all[:, n], color=node_colors[n], alpha=0.6, linewidth=1.5)
            
        ax_pred.set_title('Global View: All Nodes Prediction (Concatenated)', fontsize=14, fontweight='bold')
        ax_pred.set_ylabel('Value', fontsize=12)
        ax_pred.grid(True, alpha=0.3)
        ax_pred.spines['top'].set_visible(False)
        ax_pred.spines['right'].set_visible(False)
        
        # 3. 绘制所有节点的真实值总图
        ax_gt = axes[N+1]
        for n in range(N):
            ax_gt.plot(t_full[:, n], color=node_colors[n], alpha=0.6, linewidth=1.5)
            
        ax_gt.set_title('Global View: All Nodes Ground Truth', fontsize=14, fontweight='bold')
        ax_gt.set_ylabel('Value', fontsize=12)
        ax_gt.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax_gt.grid(True, alpha=0.3)
        ax_gt.spines['top'].set_visible(False)
        ax_gt.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(save_dir, f'sample_{b}_epoch_{epoch}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

def plot_metrics_for_run(log_file_path, run_number):
    """
    从日志文件中提取指定run的性能指标并绘制折线图
    
    参数:
    log_file_path: 日志文件路径
    run_number: 要绘制的run编号 (1-20)
    """
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 12})
    
    # 读取日志文件
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 定义正则表达式模式
    run_pattern = rf'\*{{36}}runs:{run_number}\*{{36}}'
    metric_pattern = r'index:\d+, epoch:(\d+), loss: [\d.]+, precision:([\d.]+), recall:([\d.]+), F1:([\d.]+), accuracy:([\d.]+), SHD:([\d.]+)'
    
    # 找到指定run的起始位置
    run_match = re.search(run_pattern, content)
    if not run_match:
        print(f"未找到 run {run_number}")
        return
    
    # 找到下一个run的起始位置或文件结尾
    next_run_pattern = r'\*{36}runs:\d+\*{36}'
    next_run_matches = list(re.finditer(next_run_pattern, content[run_match.end():]))
    
    if next_run_matches:
        end_pos = run_match.end() + next_run_matches[0].start()
    else:
        # 找到最终统计信息的位置
        final_stats_match = re.search(r'precision: [\d.]+', content[run_match.end():])
        if final_stats_match:
            end_pos = run_match.end() + final_stats_match.start()
        else:
            end_pos = len(content)
    
    # 提取该run的内容
    run_content = content[run_match.start():end_pos]
    
    # 提取所有指标
    metrics = re.findall(metric_pattern, run_content)
    
    if not metrics:
        print(f"run {run_number} 中未找到指标数据")
        return
    
    # 整理数据
    epochs = []
    precision = []
    recall = []
    f1 = []
    accuracy = []
    shd = []
    
    for metric in metrics:
        epochs.append(int(metric[0]))
        precision.append(float(metric[1]))
        recall.append(float(metric[2]))
        f1.append(float(metric[3]))
        accuracy.append(float(metric[4]))
        shd.append(float(metric[5]))
    
    # 设置配色方案
    colors = {
        'accuracy': '#2E86AB',   # 深蓝色
        'precision': '#A23B72',  # 紫红色
        'recall': '#F18F01',     # 橙色
        'f1': '#C73E1D',         # 红色
        'shd': '#6A994E'         # 绿色
    }
    
    # 绘制图表
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Performance Metrics for Run {run_number}', fontsize=20, fontweight='bold')
    
    # 绘制各个指标
    metrics_data = [
        (accuracy, 'Accuracy', axes[0, 0], colors['accuracy']),
        (precision, 'Precision', axes[0, 1], colors['precision']),
        (recall, 'Recall', axes[0, 2], colors['recall']),
        (f1, 'F1 Score', axes[1, 0], colors['f1']),
        (shd, 'SHD', axes[1, 1], colors['shd'])
    ]
    
    for data, title, ax, color in metrics_data:
        # 绘制主线条
        ax.plot(epochs, data, color=color, linewidth=2.5, marker='o', 
                markersize=6, markerfacecolor='white', markeredgecolor=color, 
                markeredgewidth=2, alpha=0.9, label=title)
        
        # 添加阴影效果
        ax.fill_between(epochs, data, alpha=0.2, color=color)
        
        # 设置标签和标题
        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel(title, fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', color=color)
        
        # 美化网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_facecolor('#F8F9FA')
        
        # 设置边框
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(1.5)
        
        # 添加最大值和最小值标注
        if title != 'SHD':  # SHD越小越好
            max_idx = np.argmax(data)
            ax.annotate(f'Best: {data[max_idx]:.3f}', 
                       xy=(epochs[max_idx], data[max_idx]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=11, fontweight='bold',
                       color='darkgreen',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='darkgreen', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))
        else:
            min_idx = np.argmin(data)
            ax.annotate(f'Best: {data[min_idx]:.3f}', 
                       xy=(epochs[min_idx], data[min_idx]),
                       xytext=(10, -10), textcoords='offset points',
                       fontsize=11, fontweight='bold',
                       color='darkgreen',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='darkgreen', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))
        
        # 添加图例
        ax.legend(loc='best', fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    
    # 隐藏第6个子图
    axes[1, 2].axis('off')
    
    # 在第6个子图位置添加统计信息
    stats_text = f"Run {run_number} Statistics\n" + "="*25 + "\n\n"
    stats_text += f"Total Epochs: {epochs[-1]}\n\n"
    stats_text += f"Final Values:\n"
    stats_text += f"   • Accuracy: {accuracy[-1]:.3f}\n"
    stats_text += f"   • F1 Score: {f1[-1]:.3f}\n"
    stats_text += f"   • Recall: {recall[-1]:.3f}\n"
    stats_text += f"   • Precision: {precision[-1]:.3f}\n"
    stats_text += f"   • SHD: {shd[-1]:.3f}\n\n"
    stats_text += f"Best Performance:\n"
    stats_text += f"   • Accuracy: {max(accuracy):.3f} (Epoch {epochs[np.argmax(accuracy)]})\n"
    stats_text += f"   • F1 Score: {max(f1):.3f} (Epoch {epochs[np.argmax(f1)]})\n"
    stats_text += f"   • Recall: {max(recall):.3f} (Epoch {epochs[np.argmax(f1)]})\n"
    stats_text += f"   • Precision: {max(precision):.3f} (Epoch {epochs[np.argmax(f1)]})\n"
    stats_text += f"   • SHD: {min(shd):.3f} (Epoch {epochs[np.argmin(shd)]})"
    
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=13, verticalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='#E8F4F8', 
                             edgecolor='#2E86AB', linewidth=2, alpha=0.9),
                    family='monospace')
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(os.path.split(log_file_path)[0], f'runs_{run_number}.png'))



