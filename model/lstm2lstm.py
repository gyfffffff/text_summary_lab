import torch
from torch import nn



class Encoder(nn.Module):
    def __init__(self, encoder_embedding_num, encoder_hidden_num, vocab_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, encoder_embedding_num)  
        self.lstm = nn.LSTM(encoder_embedding_num, encoder_hidden_num, batch_first=True, num_layers=1, bidirectional=True)

    def forward(self, x):
        x_emb = self.embedding(x)
        _, encoder_hidden = self.lstm(x_emb)
        return encoder_hidden
    
class Decoder(nn.Module):
    def __init__(self, decoder_embedding_num, decoder_hidden_num, vocab_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, decoder_embedding_num)
        self.lstm = nn.LSTM(decoder_embedding_num, decoder_hidden_num, batch_first=True, num_layers=num_layers)
    
    def forward(self, x, encoder_hidden):
        x_emb = self.embedding(x)
        # h, c = encoder_hidden
        decoder_output, decoder_hidden = self.lstm(x_emb, encoder_hidden)
        return decoder_output, decoder_hidden

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

        self.encoder = Encoder(encoder_emb_num, encoder_hidden_num, vocab_size, num_layers)
        self.decoder = Decoder(decoder_emb_num, decoder_hidden_num, vocab_size, num_layers)
        self.classifier = nn.Linear(decoder_hidden_num, vocab_size)
        self.cross_loss = nn.CrossEntropyLoss()
        self.BOS = 1300
        self.PAD = 1301
        self.EOS = 1302

    def forward(self, x, y):
        decoder_input = y[:, :-1]   #[batchsize, batch_max_len]
        decoder_target = y[:, 1:]   #[batchsize, batch_max_len]
        encoder_hidden = self.encoder(x)  # [2, batchsize, hidden_size]
        decoder_output, _ = self.decoder(decoder_input, encoder_hidden)   # [batchsize, batch_max_len-1, hidden_size]  给decoder多少输入，就有多少输出

        pre = self.classifier(decoder_output)  # [batchsize, batch_max_len-1, vocab_size]
        loss = self.cross_loss(pre.reshape(-1, pre.shape[-1]), decoder_target.reshape(-1))

        return loss
    
    def summary(self, x):
        result = []
        batch_size = x.shape[0]
        encoder_hidden = self.encoder(x)
        decoder_input = torch.tensor([[self.BOS]]*batch_size, device=self.device)  # [batchsize, 1]
        decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_hidden)  # [batchsize, 1, hidden_size]
        for i in range(100):
            pre = self.classifier(decoder_output)  # [batchsize, 1, vocab_size] 得到每条序列的下一个字
            pre = pre.argmax(dim=-1)  
            result.append(pre)   # 加入数组，最后会用concat
            decoder_input = pre
        result = torch.cat(result, dim=1).detach().cpu().tolist()
        # 对结果整理，去掉pad和eos
        for res in result:
            for j in range(100):
                if res[j] == self.PAD or res[j] == self.EOS:
                    res = res[:j]   # 截断
                    break
        return result

