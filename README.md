# 任务二：基于深度学习的文本分类

## 一、实验概述

### 1.1 实验背景

本实验用Pytorch重写任务一，使用CNN、RNN实现对电影评论的情感分析。

### 1.2 实验目标

（1）利用pytorch重写任务一，实现CNN、RNN的文本分类

（2）了解词嵌入、CNN/RNN的特征提取、Dropout等知识

## 二、实验设计与实现

### 2.1 数据预处理

加载数据和标签，并加载GloVe词向量，为平衡类别比重，引入类别权重，强制模型平等对待所有类别，提升整体泛化能力。利用train_test_split函数以7：3的比例划分数据集，在获得嵌入时实现。

```
    def data_split(data, labels, test_rate = 0.3):
        # 划分数据集
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=test_rate, random_state=21
        )
        # 返回训练集和测试集
        return x_train, x_test, y_train, y_test

    def load_data(data_path: str) -> tuple[list[str], np.ndarray]:
        df = pd.read_csv(data_path, sep=",", encoding="utf-8")
        return df["Phrase"].to_list(), df["Sentiment"].astype(int).values        

    def load_glove(glove_path: str = "glove.6B.50d.txt")->dict[str,list[float]]:
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

    # 加载数据和标签
    data, labels = load_data(CONFIG['data_path'])

    # 计算类别权重
    class_weights = torch.tensor(
        compute_class_weight('balanced', classes=np.unique(labels), y=labels),
        dtype=torch.float32
    )

    # 加载GloVe词向量
    trained_dict = load_glove(CONFIG['glove_path'])
```

### 2.2 配置参数

```
# 配置参数
    CONFIG = {
        'iter_times': 20,
        'learning_rate': 5e-5,
        'batch_size': 64,
        'seed': 21,
        'data_path': "D:/000personal/dataset/sentiment-analysis-on-movie-reviews/train.csv",
        'glove_path': "D:/000personal/dataset/sentiment-analysis-on-movie-reviews/glove.6B.50d.txt"
    }
```

### 2.3 RNN模型

TextRNN采用双向GRU结构捕捉文本的序列特征，通过词嵌入层将单词映射为稠密向量，经GRU层提取时序信息后，取最后一个时间步的输出并通过LayerNorm稳定训练，最终由全连接层分类。支持预训练词向量初始化，并加入Dropout防止过拟合，适合处理长距离依赖的文本数据。

```
class TextRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, num_classes=5, 
                 weight=None, num_layers=2, batch_first=True, dropout=0.3):
        """
        初始化 TextRNN 类。
        Args:
        embedding_dim (int): 词向量的维度。
        hidden_size (int): RNN 隐藏层的维度，决定记忆容量。
        vocab_size (int): 词汇表的大小。
        num_classes (int, optional): 分类的类别数,默认为5。
        weight (torch.Tensor, optional): 预训练的词向量权重，默认为 None。
        num_layers (int, optional): RNN 层的数量,默认为2。
        batch_first (bool, optional): 输入和输出是否以 batch 为第一维度，默认为 True。
        dropout (float, optional): RNN 层的 dropout 比例,默认为0.3。
        Returns:
        None
        """
        super(TextRNN, self).__init__()
        # 初始化词向量维度
        self.embedding_dim = embedding_dim
        # 初始化RNN隐藏层维度：决定记忆容量
        self.hidden_size = hidden_size
        # 初始化RNN层数
        self.num_layers = num_layers

        # 如果预训练权重为 None，则使用 Xavier 初始化方法生成预训练权重
        if weight is None:
            # 使用 xavier_normal_ 初始化方法生成预训练权重
            weight = nn.init.xavier_normal_(torch.Tensor(vocab_size, embedding_dim))
        weight = F.normalize(weight, p=2, dim=1)  # 对预训练权重进行归一化处理
        # 使用预训练权重初始化 Embedding 层
        self.embedding = nn.Embedding.from_pretrained(weight)

        # 初始化 GRU 核心
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # batch_first=True 输入和输出都是以batch为第一维度
            batch_first=batch_first,
            # 如果层数大于 1，则使用 dropout，否则不使用
            dropout=dropout if num_layers > 1 else 0, 
            bidirectional=True
            )

        # 初始化输出投影层
        self.ln = nn.LayerNorm(hidden_size*2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size*2, num_classes)
        )

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (list of int): 输入的文本序列，每个元素代表一个单词的索引。

        Returns:
            Tensor: 经过模型处理后的输出，形状为 (batch_size, num_classes)，其中 num_classes 是分类的数量。

        """
        # 输入处理
        device = next(self.parameters()).device
        x = torch.LongTensor(x).to(device)

        # 嵌入层
        embedded = self.embedding(x)

        # GRU处理
        gru_out, _ = self.gru(embedded)  # gru_out: [batch, seq_len, hid_dim*2]

        # 取最后一个时间步 + 层归一化
        last_hidden = self.ln(gru_out[:, -1, :])

        # 分类
        return self.classifier(last_hidden)
```

### 2.4 CNN模型

TextCNN使用多尺度一维卷积核（2,3,4,5）并行提取文本的局部特征，通过AdaptiveAvgPool压缩序列维度后拼接多尺度特征，经Dropout层正则化后由全连接层分类。支持预训练词向量，擅长捕捉关键词和短语模式，对短文本分类效果显著。

```
class TextCNN(nn.Module):
    def __init__(self, embedding_dim, vocab_size, max_seq_len, 
                 num_classes=5, weight=None, dropout=0.5):
        """
        Args:
            embedding_dim (int): 嵌入层的维度。
            vocab_size (int): 词汇表的大小。
            max_seq_len (int): 最大序列长度。
            num_classes (int, optional): 类别数量,默认为5。
            weight (optional): 嵌入层的权重。
            dropout (float, optional): Dropout 的比例,默认为0.5。

        Returns:
            None
        """

        # 调用父类的构造函数
        super(TextCNN, self).__init__()

        # 设置嵌入层的维度
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_filters = max_seq_len  

        # 嵌入层
        # 如果预训练的嵌入存在，则使用预训练的嵌入，否则使用默认的嵌入层
        if weight is not None:
            weight = F.normalize(weight, p=2, dim=1)
        self.embedding = nn.Embedding.from_pretrained(weight) if weight is not None \
            else nn.Embedding(vocab_size, embedding_dim)

        # 动态过滤器大小配置
        filter_sizes = [2, 3, 4, 5]
        self.convs = nn.ModuleList([
            nn.Sequential(
                # 卷积层
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=self.num_filters,
                    kernel_size=k,
                    padding=k-1
                ),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            ) for k in filter_sizes
        ])

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            # 全连接层
            nn.Linear(len(filter_sizes) * self.num_filters, num_classes)
        )

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (list of int): 输入数据，列表形式，包含多个整数。

        Returns:
            torch.Tensor: 前向传播后的输出，通常用于分类或其他下游任务。

        """
        # 输入处理
        device = next(self.parameters()).device
        if not isinstance(x, torch.Tensor):
            # 如果输入不是Tensor，则转换为Tensor
            x = torch.LongTensor(x).to(device)

        # 嵌入层
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        # 调整维度顺序 [batch_size, seq_len, embedding_dim] → [batch_size, embedding_dim, seq_len]
        x = x.permute(0, 2, 1)

        # 卷积层处理
        features = []
        for conv in self.convs:
            # 卷积输出 [batch_size, num_filters, 1]
            conv_out = conv(x)
            features.append(conv_out.squeeze(2))  # 移除最后维度

        # 合并特征 [batch_size, num_filters * len(filter_sizes)]
        concatenated = torch.cat(features, dim=1)

        return self.classifier(concatenated)
```

## 三、实验结果

（1）CNN/RNN + GloVe的测试集表现优于随机初始化，验证了预训练词向量能提升模型泛化能力。

（2）CNN + GloVe的准确率维持在0.43左右，RNN的准确率震荡明显（但已经改用GUR和添加梯度剪裁），可能需要进一步梯度控制和残差连接等。

（3）学习率较小（再提高则影响RNN效果），各模型损失逐步缓慢下降。

由于设备限制，只设置了20轮训练次数，进一步优化超参数可能提升模型效果。

![comparison_plot](D:\VSWorkSpace\Python\git-task2\comparison_plot.jpg)
