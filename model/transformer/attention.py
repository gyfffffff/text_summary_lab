import torch
import torch.nn as nn
import math


NEG_INF = float("-inf")


class ScaledDotProductAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_query_key: int,
                 d_value: int):
        """
        :param d_model: 输入的特征向量的长度
        :param d_query_key: 键以及询问的向量表示的长度
        :param d_value: 值的向量表示的长度
        """
        super(ScaledDotProductAttention, self).__init__()
        self.wk = nn.Parameter(torch.zeros((d_model, d_query_key)))
        self.wq = nn.Parameter(torch.zeros((d_model, d_query_key)))
        self.wv = nn.Parameter(torch.zeros((d_model, d_value)))
        self.div = math.sqrt(d_query_key)  # 储存以加速
        self.softmax = nn.Softmax(dim=2)

    def forward(self,
                mask: torch.Tensor,
                x_key_value: torch.Tensor,
                x_query: torch.Tensor=None):
        """
        :param mask: 一个 bool Tensor，与博客中不同的是它为 true 的地方是表示允许分配注意力，而为 false 的地方是不允许分配注意力。
        :param x_key_value: 键值对
        :param x_query: 询问
        :return: 询问中每个词询问的结果，是value向量的加权和
        """
        if x_query == None:
            x_query = x_key_value   # x_key_value [batch_size, seq_len, d_model]
        k = x_key_value @ self.wk
        q = x_query @ self.wq
        v = x_key_value @ self.wv
        mat = torch.einsum("nik, njk -> nij", q, k) / self.div
        mat = torch.where(mask, mat, NEG_INF)  # 这里 true 是通过，false 的地方设为 -inf 则在 softmax 后为 0
        portion = self.softmax(mat)
        return torch.einsum("nij, njk -> nik", portion, v)  # 计算注意力权重和 V的点积，得到加权和
