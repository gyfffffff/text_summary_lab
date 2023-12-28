import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    利用Pytorch生成指定长度的原始词嵌入
    """
    def __init__(self, vocab_size, embed_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size

    def forward(self, x):
        return self.embedding(x.long()) * math.sqrt(self.embed_size)  # 乘以sqrt(d_model)是为了与Position Embedding相加后的数值大小相近


class PositionEmbedding(nn.Module):
    """
    生成至多包含max_num_words个词的Position Embedding
    """
    def __init__(self, d_model, max_num_words=1000, p_drop=0.1):
        super(PositionEmbedding, self).__init__()
        # 计算Position Embedding，储存在 self.additive中
        temp:torch.Tensor = torch.arange(d_model) // 2 * 2 / d_model
        temp = torch.pow(10000, -temp)
        additive = torch.arange(max_num_words).view(-1, 1)
        additive = additive.repeat(1, d_model)
        additive = additive / temp
        additive[:, 0::2] = torch.sin(additive[:, 0::2])
        additive[:, 1::2] = torch.sin(additive[:, 1::2])
        self.additive = nn.Parameter(additive, requires_grad=False)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        len_x = x.size()[1]
        # 截取所需长度的Position Embedding，并通过Dropout
        return self.dropout(self.additive[:len_x] + x)
