import torch
import torch.nn as nn
import torch.nn.functional as F

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