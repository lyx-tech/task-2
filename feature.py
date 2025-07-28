import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch


def data_split(data, labels, test_rate = 0.3):
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_rate, random_state=21
    )
    # 返回训练集和测试集
    return x_train, x_test, y_train, y_test


class RandomEmbedding():
    #随机初始化
    def __init__(self, data, labels, test_rate = 0.3):
        # 单词到ID的映射
        self.dict_words = {'<PAD>': 0,'<UNK>': 1}
        # 按照句子长度排序，短的在前
        data.sort(key = lambda x: len(x.split())) 
        self.data = data
        self.len_words = 0
        # 分割数据集为训练集和测试集
        self.train, self.test ,self.train_y, self.test_y= data_split(data, labels, test_rate = test_rate)
        # 训练集的单词ID列表，叠成一个矩阵
        self.train_matrix = list()  
        # 测试集的单词ID列表，叠成一个矩阵
        self.test_matrix = list()   
        self.longest = 0
        
    def get_words(self):
        for term in self.data:
            term = term.upper()   #转为大写
            words = term.split()  #分成单词
            for word in words:
                if word not in self.dict_words:
                    # 新单词从2开始编号，0是padding，1是未知词
                    self.dict_words[word] = len(self.dict_words)
        self.len_words = len(self.dict_words)  #单词数目
        
    def get_id(self):
        # 训练集
        for term in self.train:
            # 将term转换为大写
            term = term.upper()
            # 将term按空格分割成单词列表
            words = term.split()
            # 使用列表推导式将单词列表中的每个单词映射为字典中的索引，生成新的单词索引列表
            item = [self.dict_words[word] for word in words] 
            # 记录最长句子的单词数
            self.longest = max(self.longest, len(item))  
            # 将生成的单词索引列表添加到训练集矩阵中
            self.train_matrix.append(item)

        # 测试集
        for term in self.test:
            # 将term转换为大写
            term = term.upper()
            # 将term按空格分割成单词列表
            words = term.split()
            # 对于未知词，返回字典中不存在的键对应的默认值1
            # 未知词返回1
            item = [self.dict_words.get(word,1) for word in words]
            # 记录最长句子的单词数
            self.longest = max(self.longest, len(item))
            # 将生成的单词索引列表添加到测试集矩阵中
            self.test_matrix.append(item)

        # 更新单词总数
        self.len_words += 1
        
class GloveEmbedding():
    def __init__(self, data, labels, trained_dict, test_rate=0.3):
        self.dict_words = {'<PAD>': 0, '<UNK>': 1}
        self.trained_dict = trained_dict
        self.data = sorted(data, key=lambda x: len(x.split()))
        self.train, self.test, self.train_y, self.test_y = train_test_split(
            self.data, labels, test_size=test_rate, random_state=21
        )
        self.embedding = self.__init_embedding()
        self.train_matrix = []
        self.test_matrix = []
        self.longest = 0
        self.len_words = 2

    def __init_embedding(self):
        return [
            [0.0] * 50,  # <PAD>
            np.random.normal(scale=0.1, size=50).tolist()  # <UNK>
        ]

    def get_words(self):
        for word in ['<PAD>', '<UNK>']:  # 确保特殊符号存在
            if word not in self.dict_words:
                self.dict_words[word] = len(self.dict_words)

        for term in self.data:
            for word in term.upper().split():
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words)
                    vec = self.trained_dict.get(word, self.embedding[1])
                    self.embedding.append(vec)

        # 归一化处理
        self.embedding = [
            vec if i == 0 else (np.array(vec) / (np.linalg.norm(vec) + 1e-8)).tolist()
            for i, vec in enumerate(self.embedding)
        ]
        self.len_words = len(self.dict_words)

    def get_id(self, max_len=128):
        for term in self.train:
            words = term.upper().split()[:max_len]
            self.train_matrix.append([self.dict_words.get(w, 1) for w in words])
            self.longest = max(self.longest, len(words))
        
        for term in self.test:
            words = term.upper().split()[:max_len]
            self.test_matrix.append([self.dict_words.get(w, 1) for w in words])
            self.longest = max(self.longest, len(words))

class TextClDataset(Dataset):
    #自定义数据集的结构
    def __init__(self, sentence, emotion):
        # 句子序列
        self.sentence = sentence  
        # 情感标签  
        self.emotion = emotion    
        
    def __getitem__(self, item):
        # 返回单个样本句子和对应的情感标签
        return self.sentence[item], self.emotion[item]
    
    def __len__(self):
        # 返回数据集大小
        return len(self.emotion)
    
    
def pad_collate(batch_data):
    # 按照句子长度排序，长的在前
    batch_data.sort(key = lambda x: len(x[0]), reverse = True)  
    # 分批次，并进行padding
    # 解压：将数据分离为句子列表和标签列表
    sentence, emotion = zip(*batch_data)
    # 将句子列表转换为Longtensor类型
    sentences = [torch.LongTensor(sent) for sent in sentence]  
    # 将所有句子填充到相同长度
    # batch_first=True表示batch的维度在第一位，padding_value=0表示填充值为0
    padded_sents = pad_sequence(sentences, batch_first = True, padding_value = 0) 
    return torch.LongTensor(padded_sents), torch.LongTensor(emotion)

def get_batch(x, y, batch_size):
    #利用dataloader划分batch
    dataset = TextClDataset(x, y)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True, collate_fn = pad_collate)
    return dataloader