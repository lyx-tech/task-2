import csv
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import torch
from visualization import train_model, plot_results
from cnn_rnn import TextCNN, TextRNN
from feature import get_batch, GloveEmbedding, RandomEmbedding

def load_data(data_path: str) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(data_path, sep=",", encoding="utf-8")
    return df["Phrase"].to_list(), df["Sentiment"].astype(int).values

def load_glove(glove_path: str = "glove.6B.50d.txt") -> dict[str, list[float]]:
    glove_dict = {"<UNK>": [0.0] * 50}
    try:
        with open(glove_path, 'rb') as f:
            for line in f:
                parts = line.split()
                word = parts[0].decode('utf-8').upper()
                glove_dict[word] = [float(val) for val in parts[1:51]]
    except FileNotFoundError:
        raise FileNotFoundError(f"Glove file not found at {glove_path}")
    return glove_dict

def main():
    # 配置参数
    CONFIG = {
        'iter_times': 20,
        'learning_rate': 5e-5,
        'batch_size': 64,
        'seed': 21,
        'data_path': "D:/000personal/dataset/sentiment-analysis-on-movie-reviews/train.csv",
        'glove_path': "D:/000personal/dataset/sentiment-analysis-on-movie-reviews/glove.6B.50d.txt"
    }

    # 设置随机种子
    torch.manual_seed(CONFIG['seed'])
    random.seed(CONFIG['seed'])
   
    # 加载数据和标签
    data, labels = load_data(CONFIG['data_path'])

    # 计算类别权重
    class_weights = torch.tensor(
        compute_class_weight('balanced', classes=np.unique(labels), y=labels),
        dtype=torch.float32
    )

    # 加载GloVe词向量
    trained_dict = load_glove(CONFIG['glove_path'])

    # 创建随机嵌入模型
    random_embedding = RandomEmbedding(data=data, labels=labels)
    random_embedding.get_words()  # 获取单词
    random_embedding.get_id()  # 获取单词ID
    
    # 创建GloVe嵌入模型
    glove_embedding = GloveEmbedding(data=data, labels=labels, trained_dict=trained_dict)
    glove_embedding.get_words()  # 获取单词
    glove_embedding.get_id()  # 获取单词ID
    
    # 初始化结果字典
    results = {
        model_type: {
            metric: [] for metric in ['train_loss', 'test_loss', 'long_loss', 'train_acc', 'test_acc', 'long_acc']
        } for model_type in ['RNN+random', 'CNN+random', 'RNN+glove', 'CNN+glove']
    }

    # 名称映射
    name_mapping = {
        'random_rnn': 'RNN+random',
        'random_cnn': 'CNN+random',
        'glove_rnn': 'RNN+glove',
        'glove_cnn': 'CNN+glove'
    }

    # 遍历不同模型和嵌入方式
    for name, embedding, use_glove in [
        ('random_rnn', random_embedding, False),
        ('random_cnn', random_embedding, False),
        ('glove_rnn', glove_embedding, True),
        ('glove_cnn', glove_embedding, True)
    ]:
        # 检查名称是否在映射中
        if name not in name_mapping:
            continue
        
        # 获取训练和测试数据批次
        train_data = get_batch(embedding.train_matrix, embedding.train_y, CONFIG['batch_size'])
        test_data = get_batch(embedding.test_matrix, embedding.test_y, CONFIG['batch_size'])
        
        # 创建模型实例
        model_class = TextRNN if 'rnn' in name else TextCNN
        model_args = (50, 50, embedding.len_words) if 'rnn' in name else (50, embedding.len_words, embedding.longest)
        model = model_class(*model_args, weight=torch.tensor(embedding.embedding, dtype=torch.float) if use_glove else None)

        # 训练模型并获取指标
        metrics = train_model(model, train_data, test_data, CONFIG['learning_rate'], CONFIG['iter_times'], class_weights)
        results[name_mapping[name]] = metrics

    # 绘制结果图
    plot_results(results, CONFIG['iter_times'])
    
if __name__ == "__main__":
    main()