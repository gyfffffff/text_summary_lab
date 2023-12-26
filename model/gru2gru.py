"""
    1. encoder 使用单层双向 GRU, 最后一层的输出不是拼接，而是相加
    2. decoder 使用双层单项 GRU，第一个字符<bos>输入的hidden使用encoder 端的输出初始化的
    3. 为了架构清晰，encoder 和 decoder 的 hidden_siz 取相同值，并且取 hidden_siz = embed_size
""" 
import random
import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1):
        super(Encoder, self).__init__()

        self.input_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, inputs):
        # 输出维度:[batch, seq, embed_size]
        embeddings = self.embedding(inputs)
        _, hidden = self.gru(embeddings)
        hidden = torch.cat(hidden[0, :, :], hidden[1, :, :], dim=1)  # 可以试试相加
        hidden = torch.tanh(self.fc(hidden)).unsqueeze(0)
        return hidden
class Atten(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(Atten, self).__init__()
        self.enc_hidden_size = encoder_hidden_size
        self.dec_hidden_size = decoder_hidden_size
        self.fc_in = nn.Linear(
            encoder_hidden_size*2, decoder_hidden_size, bias=False)
        self.fc_out = nn.Linear(
            encoder_hidden_size*2 + decoder_hidden_size, decoder_hidden_size)

    def forward(self, output, context):
        # output [batch, target_len, dec_hidden_size]
        # context [batch, source_len, enc_hidden_size*2]
        batch_size = output.size(0)
        y_len = output.size(1)
        x_len = context.size(1)
        # [batch_size * x_sentence_len, enc_hidden_size*2]
        x = context.contiguous().view(batch_size*x_len, -1)
        # [batch_size * x_len, dec_hidden_size]
        x = self.fc_in(x)
        # [batch_size, x_sentence_len, dec_hidden_size]
        context_in = x.view(batch_size, x_len, -1)
        # [batch_size, y_sentence_len, x_sentence_len]
        atten = torch.bmm(output, context_in.transpose(1, 2))
        # [batch_size, y_sentence_len, x_sentence_len]
        atten = F.softmax(atten, dim=2)
        # [batch_size, y_sentence_len, enc_hidden_size*2]
        context = torch.bmm(atten, context)
        # [batch_size, y_sentence_len, enc_hidden_size*2+dec_hidden_size]
        output = torch.cat((context, output), dim=2)
        # [batch_size * y_sentence_len, enc_hidden_size*2+dec_hidden_size]
        output = output.contiguous().view(batch_size*y_len, -1)
        output = torch.tanh(self.fc_out(output))
        # [batch_size, y_sentence_len, dec_hidden_size]
        output = output.view(batch_size, y_len, -1)
        return output, atten
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=2):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, dec_hidden):
        embeddings = self.embedding(inputs)
        output, _ = self.gru(embeddings, dec_hidden)
        output = self.out(output.squeeze())
        return output
    
class Seq2Seq(nn.Module):
    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        encoder_emb_num = args['embed_size']
        encoder_hidden_num = args['hidden_size']
        decoder_emb_num = args['embed_size']
        decoder_hidden_num = args['hidden_size']
        vocab_size = args['vocab_size']
        num_layers = args['num_layers']
        self.device = args['device']
        self.batch_size = args['batch_size']
        self.BOS = 1300
        self.PAD = 1301
        # self.EOS = 1302


        self.encoder = Encoder(encoder_emb_num, encoder_hidden_num, vocab_size, num_layers)
        self.decoder = Decoder(decoder_emb_num, decoder_hidden_num, vocab_size, num_layers)

    def forward(self, X, Y, teaching_rate=0.5):
        loss = 0
        loss_func = nn.CrossEntropyLoss()
        num_not_pad_tokens = 0
        encoder_hidden = self.encoder(X)
        decoder_input = torch.tensor([[self.BOS]]*self.batch_size, device=self.device)
        mask = torch.ones(self.batch_size, device=self.device)
        for y in Y:
            decoder_out = self.decoder(decoder_input, encoder_hidden)
            loss += (mask * loss_func(decoder_out, y)).sum()
            use_teaching = random.random() < teaching_rate
            if use_teaching:
                decoder_input = y
            else:
                decoder_input = decoder_out.argmax(dim=-1)
            num_not_pad_tokens += mask.sum().item()
            # PAD对应mask为0，否则为1
            mask = mask * (decoder_input != self.PAD).float()
        return loss / num_not_pad_tokens
        
device = 'cuda'
if __name__=='__main__':
    model = Seq2Seq({'embed_size': 128, 'hidden_size': 128, 'vocab_size': 1303, 'num_layers': 2, 'device': device, 'batch_size': 2})   
    model.to(device)
    X = torch.tensor([[1,2,3,4,5,6,7, 1301, 1301, 1302], [1,2,3,4,5,6,1301, 1301, 1301, 1302]], dtype=torch.long).to(device)
    Y = torch.tensor([[1300, 1,2,3,4,5,6,7, 1301, 1302], [1300, 4,4,5,3,2,1301, 1301, 1301, 1302]], dtype=torch.long).to(device)
    loss = model(X, Y)    
