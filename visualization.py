import matplotlib.pyplot as plt
import torch 
from torch import optim
from torch.nn import functional as F
import torch.nn as nn

def train_model(model, train_data, test_data, learning_rate, iter_times, class_weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    metrics = {
        'train_loss': [], 'test_loss': [], 'long_loss': [],
        'train_acc': [], 'test_acc': [], 'long_acc': []
    }
    
    for iteration in range(iter_times):
        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for x, y in train_data:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
        
        metrics['train_loss'].append(train_loss / len(train_data))
        metrics['train_acc'].append(train_correct / train_total)
        
        # 评估阶段
        model.eval()
        with torch.no_grad():
            for name, data, is_long in [
                ('train', train_data, False),
                ('test', test_data, False),
                ('long', test_data, True)
            ]:
                total_loss, total_correct, total_samples = 0, 0, 0
                
                for x, y in data:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = criterion(pred, y)
                    _, y_pred = torch.max(pred, -1)
                    
                    total_loss += loss.item()
                    total_correct += (y_pred == y).sum().item()
                    total_samples += y.size(0)
                    
                    if is_long and len(x[0]) > 20:
                        metrics['long_loss'].append(loss.item())
                        metrics['long_acc'].append((y_pred == y).float().mean().item())
                
                metrics[f'{name}_loss'].append(total_loss / len(data))
                metrics[f'{name}_acc'].append(total_correct / total_samples)

        print(f"Iteration {iteration+1} - Train: Loss {metrics['train_loss'][-1]:.4f}, Acc {metrics['train_acc'][-1]:.4f} | "
              f"Test: Loss {metrics['test_loss'][-1]:.4f}, Acc {metrics['test_acc'][-1]:.4f}")
    
    return metrics


'''
def plot_results(results, iter_times):
    """
    绘制训练结果。

    Args:
        results (dict): 包含各个模型在训练和测试中的性能指标。
        iter_times (int): 迭代次数。

    Returns:
        None

    """

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 定义要绘制的图形列表
    plots = [
        ('Train Loss', 'train_loss', 'Loss'),
        ('Test Loss', 'test_loss', 'Loss'),
        ('Train Accuracy', 'train_acc', 'Accuracy'),
        ('Test Accuracy', 'test_acc', 'Accuracy')
    ]
    
    # 遍历每个子图和对应的绘图数据
    for ax, (title, metric_key, ylabel) in zip(axes.flat, plots):
        # 遍历每个模型和对应的颜色及样式
        for model_name, color in [
            ('RNN+random', 'r-o'),
            ('CNN+random', 'g-s'),
            ('RNN+glove', 'b-d'),
            ('CNN+glove', 'y-*')
        ]:
            y_data = results[model_name][metric_key]
            x = list(range(1, len(y_data) + 1))  # 动态调整 x 的长度
            ax.plot(x, y_data, color, label=model_name)
            # 绘制图形
            ax.plot(x, y_data, color, label=model_name)

        # 设置网格线
        ax.grid(True, which='both', linestyle='--', linewidth=1.0, color='lightgray')
        # 设置背景颜色
        ax.set_facecolor('#f9f9f9')
        # 设置标题
        ax.set_title(title)
        # 设置x轴标签
        ax.set_xlabel('Iterations')
        # 设置y轴标签
        ax.set_ylabel(ylabel)
        # 设置图例字体大小
        ax.legend(fontsize=10)
        # 设置y轴范围
        ax.set_ylim(0, 1 if 'Accuracy' in ylabel else None)
    
    # 调整子图布局
    plt.tight_layout()
    # 保存图形
    plt.savefig('main_plot.jpg')
    # 显示图形
    plt.show()

'''
def plot_results(results, iter_times):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 定义模型样式（新增glove版本）
    model_styles = {
        'RNN+random': ('r-', 'r:'),  # (train_style, test_style)
        'CNN+random': ('g-', 'g:'),
        'RNN+glove': ('b-', 'b:'),
        'CNN+glove': ('m-', 'm:')    # 洋红色(magenta)代表CNN+glove
    }
    
    # 第一张图：损失对比
    for model_name, (train_style, test_style) in model_styles.items():
        axes[0].plot(results[model_name]['train_loss'], train_style, 
                    label=f'{model_name} (Train)', linewidth=2)
        axes[0].plot(results[model_name]['test_loss'], test_style, 
                    label=f'{model_name} (Test)', linewidth=2)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training vs Validation Loss', fontsize=14)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # 第二张图：准确率对比
    for model_name, (train_style, test_style) in model_styles.items():
        axes[1].plot(results[model_name]['train_acc'], train_style, 
                    label=f'{model_name} (Train)', linewidth=2)
        axes[1].plot(results[model_name]['test_acc'], test_style, 
                    label=f'{model_name} (Test)', linewidth=2)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_xlabel('Iterations', fontsize=12)
    axes[1].set_title('Training vs Validation Accuracy', fontsize=14)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存高清图像
    plt.savefig('comparison_plot.jpg', dpi=300, bbox_inches='tight')
    plt.show()
