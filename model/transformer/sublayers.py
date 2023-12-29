"""
sublayers.py 实现了 FFN 和 MDA。
由于所有的 sub-layer 都需要先加上 residual connection 然后再经过 LayerNorm，在这里就直接把这两个过程实现了。
"""
import torch
import torch.nn as nn
from model.transformer.attention import ScaledDotProductAttention


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, p_drop: float=0.1):
        """
        :param d_model: 在这里是上一层 MHA 输出的value向量的长度
        :param d_ffn: 中间层的长度
        :param p_drop: Dropout参数
        """
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor):
        # Linear
        additive = self.linear2(torch.relu(self.linear1(x)))
        # Dropout
        additive = self.dropout(additive)
        # Residual Connection
        x = x + additive
        # Layer Normalization
        return torch.layer_norm(x, normalized_shape=x.size()[1:])


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 p_drop: float=0.1):
        """
        :param num_heads: 博客以及原文中的 h，堆叠的 SDA 的数量
        :param d_model: 输入的特征向量长度，实际上由于 residual connection，这就是词嵌入的长度
        :param p_drop: Dropout参数
        """
        super(MultiHeadAttention, self).__init__()
        self.attentions = nn.ModuleList(
            ScaledDotProductAttention(d_model,
                                      d_model // num_heads,
                                      d_model // num_heads)
            for _ in range(num_heads)
        )  # 直接创建 h 个 SDA
        self.wo = nn.Parameter(torch.zeros((d_model, d_model)))
        self.dropout = nn.Dropout(p_drop)

    def forward(self,
                mask: torch.Tensor,
                x_key_value: torch.Tensor,
                x_query: torch.Tensor=None):
        # h x SDAs
        res = torch.concatenate([func(mask, x_key_value, x_query)
                                 for func in self.attentions], dim=2)
        # Dropout & Residual Connection
        # 对于 Encoder，只有一个输入，而 Decoder 有两个输入，residual connection 使用
        # 来自 Decoder 的输入
        res = self.dropout(res @ self.wo) \
              + x_key_value if x_query is None else x_query
        # Layer Normalization
        return torch.layer_norm(res, normalized_shape=res.size()[1:])
