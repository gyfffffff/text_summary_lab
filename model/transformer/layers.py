from sublayers import *
import torch.nn as nn
import torch


class EncoderLayer(nn.Module):
    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 d_ffn: int,
                 p_drop: float=0.1):
        """
        :param num_heads: MHA中的并行SDA数量
        :param d_model: 输入特征向量的长度，实际上就是词嵌入的长度
        :param d_ffn: FFN的隐层的维数
        :param p_drop: Dropout参数
        """
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(num_heads, d_model, p_drop)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ffn, p_drop)

    def forward(self,
                mask: torch.Tensor,
                x: torch.Tensor):
        """
        :param mask: 原文的Padding mask
        :param x: 上一层EncoderLayer的输出/原文词嵌入
        """
        x = self.attention(mask, x)
        return self.feed_forward(x)


class DecoderLayer(nn.Module):
    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 d_ffn: int,
                 p_drop: float=0.1):
        """
        :param num_heads: MHA中的并行SDA数量
        :param d_model: 输入特征向量的长度，实际上就是词嵌入的长度
        :param d_ffn: FFN的隐层的维数
        :param p_drop: Dropout参数
        """
        super(DecoderLayer, self).__init__()
        self.masked_attention = MultiHeadAttention(num_heads, d_model, p_drop)
        self.attention = MultiHeadAttention(num_heads, d_model, p_drop)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ffn, p_drop)

    def forward(self,
                padding_mask: torch.Tensor,
                mask: torch.Tensor,
                x_decoder: torch.Tensor,
                x_encoder: torch.Tensor):
        """
        :param padding_mask: 第二个MHA的mask，针对x_encoder，所以应该传入\
         mask_input
        :param mask: 第二个MHA，即mask-MHA的mask，针对 x_decoder，包含了\
         mask_output和mask_pred
        :param x_decoder: 上一层DecoderLayer的输出/译文词嵌入
        :param x_encoder: 来自Encoder的输入
        """
        x_decoder = self.masked_attention(mask, x_decoder)
        x_decoder = self.attention(padding_mask, x_encoder, x_decoder)
        return self.feed_forward(x_decoder)
