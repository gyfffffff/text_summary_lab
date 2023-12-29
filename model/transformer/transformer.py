from model.transformer.layers import *
from model.transformer.embed import *
import torch

class Encoder(nn.Module):
    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 d_model: int,
                 d_ffn: int,
                 p_drop: float=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(num_heads, d_model, d_ffn, p_drop)
            for _ in range(num_layers)
        )

    def forward(self,
                padding_mask: torch.Tensor,
                src: torch.Tensor):
        for layer in self.layers:
            src = layer(padding_mask, src)
        return src


class Decoder(nn.Module):
    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 d_model: int,
                 d_ffn: int,
                 p_drop: float=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(num_heads, d_model, d_ffn, p_drop)
            for _ in range(num_layers)
        )

    def forward(self,
                padding_mask: torch.Tensor,
                mask: torch.Tensor,
                tgt: torch.Tensor,
                encoder_out: torch.Tensor):
        for layer in self.layers:
            tgt = layer(padding_mask, mask, tgt, encoder_out)
        return tgt


class Transformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 num_heads: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int,
                 d_ffn: int,
                 p_drop: float=0.1,
                 max_num_words: int=None):
        """
        :param num_encoder_layers: EncoderLayer的数量
        :param num_decoder_layers: DecoderLayer的数量，理论上可以和EncoderLayer不同
        :param num_heads: MHA中的并行SDA数量
        :param src_vocab_size: 原文的单词库大小，用于构建pytorch自带的Embedding
        :param tgt_vocab_size: 译文的单词库大小，用于构建pytorch自带的Embedding
        :param d_model: 词嵌入的维数，由于residual connection的存在，这一维数是整个transformer中大多数特征向量的维数
        
        :param d_ffn: FFN隐层的维数
        :param p_drop: Dropout参数，一般取0.1
        :param max_num_words: 最大允许的单个sample包含的token数量，用于初始化Position Embedding
        """
        super(Transformer, self).__init__()
        self.src_tok_embed = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_tok_embed = TokenEmbedding(tgt_vocab_size, d_model)
        if max_num_words is None:
            self.pos_embed = PositionEmbedding(d_model, p_drop=p_drop)
        else:
            self.pos_embed = PositionEmbedding(d_model, max_num_words, p_drop)
        # 这里是原文的一个设计：并行的SDA中的特征向量（key,value,query）的维度都设计为
        # d_model/num_heads，从而降低计算量
        assert d_model % num_heads == 0
        self.encoder = Encoder(num_encoder_layers, num_heads, d_model, d_ffn,
                               p_drop)
        self.decoder = Decoder(num_decoder_layers, num_heads, d_model, d_ffn,
                               p_drop)
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self,
                src_padding_mask: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                src: torch.Tensor,
                tgt: torch.Tensor):
        """
        :param src_padding_mask: 原文的Padding mask
        :param tgt_padding_mask: 译文的Padding mask
        :param src: 原文（经过tokenize）
        :param tgt: 译文（经过tokenize）
        :return: 对于译文下一个词的预测scores
        """
        src_embed = self.pos_embed(self.src_tok_embed(src))  # 普通Embedding [batch_size, seq_len, d_model]
        tgt_embed = self.pos_embed(self.tgt_tok_embed(tgt))
        encoder_padding_mask = src_padding_mask.expand(-1, src.size()[1], -1)  # 新的掩码张量， -1, src.size()[1], -1 表示在第一个和第三个维度上保持 src_padding_mask 的原来大小，而在第二个维度上扩展到 src 的大小。
        encoder_out = self.encoder(encoder_padding_mask, src_embed)  # [batch_size, seq_len+3, d_model]
        decoder_mask = torch.tril(tgt_padding_mask.expand(-1, tgt.size()[1], -1))   # 下三角矩阵，因为decoder只能看到当前和之前的词
        decoder_padding_mask = src_padding_mask.expand(-1, tgt.size()[1], -1)
        decoder_out = self.decoder(decoder_padding_mask, decoder_mask,   # [batch_size, seq_len, d_model]
                                   tgt_embed, encoder_out)
        return self.linear(decoder_out)
