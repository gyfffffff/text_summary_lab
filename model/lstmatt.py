import random
import torch
from torch import nn
from torch.nn import functional as F
# 单层 双向
class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.LSTM(embed_size, hidden_size,
                          num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)  # [batch, seq_len, embed_size]
        output, hidden = self.gru(embeddings)  # output: [batch, seq_len, hidden_size*2]  hidden: [2, batch, hidden_size]
        h, c = hidden[0], hidden[1]
        h = torch.cat((h[0, :, :], h[1, :, :]), dim=1)
        c = torch.cat((c[0, :, :], c[1, :, :]), dim=1)
        h = torch.tanh(self.fc(h)).unsqueeze(0)   # [1, batch, hidden_size]
        c = torch.tanh(self.fc(c)).unsqueeze(0)
        hidden = (h, c)   # [batch, hidden_size*2]
        # hidden = torch.tanh(self.fc(hidden)).unsqueeze(0)   
        return output, hidden

    
class Atten(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(Atten, self).__init__()
        self.fc_in = nn.Linear(   # 转换输入的上下文向量
            encoder_hidden_size*2, decoder_hidden_size, bias=False)
        self.fc_out = nn.Linear(  # 合并上下文向量和输出向量
            encoder_hidden_size*2 + decoder_hidden_size, decoder_hidden_size)

    def forward(self, output, context):  # output是解码器的前一步的输出，context是编码器的输出
        batch_size = output.size(0)
        y_len = output.size(1)
        x_len = context.size(1)
        x = context.contiguous().view(batch_size*x_len, -1)
        x = self.fc_in(x)
        context_in = x.view(batch_size, x_len, -1)
        atten = torch.bmm(output, context_in.transpose(1, 2))
        atten = F.softmax(atten, dim=2)  # softmax 确保权重和为1
        context = torch.bmm(atten, context)  # 注意力权重用于计算上下文向量的加权和
        output = torch.cat((context, output), dim=2)  # 加权和与原始的output连接并通过fc_out层
        output = output.contiguous().view(batch_size*y_len, -1)
        output = torch.tanh(self.fc_out(output))
        output = output.view(batch_size, y_len, -1)
        return output, atten

# 单层 单向
class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.atten = Atten(hidden_size, hidden_size)
        self.lstm = nn.LSTM(embed_size+hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, dec_hidden, enc_outputs):
        embeddings = self.embedding(inputs)   # [batch, 1, embed_size]
        context, atten = self.atten(dec_hidden[1].transpose(0,1), enc_outputs)  # context: [batch, 1, hidden_size]  atten: [batch, 1, seq_len]
        
        embed_context = torch.cat((embeddings, context), dim=2)
        output, dec_hidden = self.lstm(embed_context, dec_hidden)
        output = self.out(output.squeeze()) # output: [batch, vocab_size]
        # print(output.shape)
        return output, dec_hidden
    
class Seq2Seq(nn.Module):
    def __init__(self, args):
        super().__init__()
        encoder_emb_num = args['embed_size']
        encoder_hidden_num = args['hidden_size']
        decoder_emb_num = args['embed_size']
        decoder_hidden_num = args['hidden_size']
        vocab_size = args['vocab_size']
        num_layers = args['num_layers']
        self.device = args['device']

        self.encoder = Encoder(encoder_emb_num, encoder_hidden_num, vocab_size)
        self.decoder = Decoder(decoder_emb_num, decoder_hidden_num, vocab_size)
        self.classifier = nn.Linear(decoder_hidden_num, vocab_size)
        self.cross_loss = nn.CrossEntropyLoss()
        self.BOS = 1300
        self.PAD = 1301
        self.EOS = 1302

    def forward(self, x, y):
        decoder_input = y[:, :-1]   #[batchsize, batch_max_len]
        decoder_target = y[:, 1:]   #[batchsize, batch_max_len]
        encoder_output, encoder_hidden = self.encoder(x)  # x: [batch_size, batch_max_len] encoder_output: [batchsize, seq_len, hidden_size*2]  encoder_hidden: [1, batchsize, hidden_size]
        loss = 0
        for i in range(decoder_input.shape[1]): # 因为要拼接，所以一个个输入
            one_input = decoder_input[:, i]  # [batchsize]
            one_target = decoder_target[:, i]  # [batchsize]
            one_output = one_input
            # teacher forcing
            one_input_copy = one_input.clone()
            for k in range(one_input_copy.shape[0]):
                if random.random() < 0.9:
                    one_input_copy[k] = one_output.argmax(-1)
            one_input = one_input_copy
            one_output, _ = self.decoder(one_input.unsqueeze(1), encoder_hidden, encoder_output)     
            loss += self.cross_loss(one_output, one_target)
        
        return loss
    
    def summary(self, x):
        result = []
        batch_size = x.shape[0]
        encoder_out, encoder_hidden = self.encoder(x)
        decoder_input = torch.tensor([[self.BOS]]*batch_size, device=self.device)  # [batchsize, 1]
        decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_hidden, encoder_out)  # [batchsize, 1, hidden_size]
        # pre = self.classifier(decoder_output)
        # pre = pre.argmax(dim=-1)
        # for one_output in decoder_output:
        for i in range(100):
            pre = decoder_output
            pre = pre.argmax(dim=-1)  
            result.append(pre.unsqueeze(1))   # 加入数组，最后会用concat
            # if pre.item() == self.PAD or pre.item() == self.EOS or len(result) > 100:
            #     break
            # result.append(pre.item())
            decoder_input = pre
        result = torch.cat(result, dim=1).detach().cpu().tolist()
        # 对结果整理，去掉pad和eos
        for res in result:
            for j in range(100):
                if res[j] == self.PAD or res[j] == self.EOS:
                    res = res[:j]   # 截断
                    break
        return result
