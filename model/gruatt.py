# %%
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as Data
import random
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import matplotlib.pyplot as plt

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


# %%
# 构建词典
UNK, PAD, BOS, EOS = '<unk>', '<pad>', '<bos>', '<eos>'
def build_vocab(descriptions, summaries):
    all_content = []
    all_content.extend(descriptions)
    all_content.extend(summaries)
    vocab = {}
    id = 0
    for sent in all_content:
        for token in sent:
            if token not in vocab:
                vocab[token] = id
                id += 1
    vocab.update({UNK: len(vocab), PAD: len(vocab)+1,
                 BOS: len(vocab)+2, EOS: len(vocab)+3})
    return vocab


# %%
# 根据构建的词典，将字符转换为对应的索引
def word2index(vocab, processed_text, max_len, type='summary'):
    content = []
    for sent in processed_text:
        if type == 'summary':
            # 如果是summary，则要加EOS和BOS
            if len(sent) < (max_len-1):
                sent.extend([EOS] + [PAD] * (max_len-len(sent)-1))
            else:
                sent = sent[:max_len-1] + [EOS]
        else:  # 如果是 description
            if len(sent) < max_len:
                sent.extend([PAD] * (max_len - len(sent)))
            else:
                sent = sent[:max_len]
        sent_idx = []
        for word in sent:
            # 转换为index
            idx = vocab.get(word, vocab[UNK])
            sent_idx.append(idx)
        content.append(sent_idx)
    return torch.tensor(content)


# %%
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.input_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size,
                          n_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.hidden_size*2, self.hidden_size)

    def forward(self, inputs, init_hidden):
        # embedding输出[batch, seq, embed_size]
        embeddings = self.embedding(inputs)
        outputs, hidden = self.gru(embeddings, init_hidden)
        hidden = torch.cat((hidden[0, :, :], hidden[1, :, :]), dim=1)
        hidden = torch.tanh(self.fc(hidden)).unsqueeze(0)
        return outputs, hidden

    def get_init_hidden(self):
        return None

# %%
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


# %%
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.atten = Atten(hidden_size, hidden_size)
        self.gru = nn.GRU(embed_size+hidden_size, hidden_size, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, dec_hidden, enc_outputs):
        embeddings = self.embedding(inputs)
        context, atten = self.atten(dec_hidden.transpose(0,1), enc_outputs)
        # output: [batch, vocab_size]
        embed_context = torch.cat((embeddings, context), dim=2)
        output, dec_hidden = self.gru(embed_context, dec_hidden)
        output = self.out(output.squeeze())
        # print(output.shape)
        return output, dec_hidden

    # 使用encoder输出的hidden作为decoder隐状态的初始化
    def get_init_hidden(self, enc_hidden):
        return enc_hidden

# %%
def batch_loss(encoder, decoder, X, Y, vocab, loss, teaching_ratio=0.5):
    # X: encoder端的输入
    # Y: decoder端的输入
    batch_size = X.shape[0]
    l = torch.tensor([0.0]).to(device)
    init_hidden = encoder.get_init_hidden()
    enc_outputs, enc_state = encoder(X, init_hidden)
    # 初始化decoder的隐藏状态
    dec_state = decoder.get_init_hidden(enc_state)
    # decoder在最初时间步的输入是BOS
    dec_input = torch.tensor([vocab[BOS]] * batch_size).to(device)
    # 使用mask来忽略掉标签为填充项PAD的损失
    mask = torch.ones(batch_size).to(device)
    num_not_pad_tokens = 0
    # Y: (batch, seq_len)
    for y in Y.permute(1, 0):
        # print(dec_input.shape)
        dec_output, dec_state = decoder(dec_input.unsqueeze(1), dec_state, enc_outputs)
        # print(dec_output.shape)
        l = l + (mask * loss(dec_output, y)).sum()
        use_teaching = random.random() < teaching_ratio
        # dec_input = y
        if use_teaching:
            # 使用 ground_truth
            dec_input = y
        else:
            # dec_output: [batch, vocab_size]
            topv, topi = dec_output.topk(1)
            dec_input = topi.squeeze().detach()
        num_not_pad_tokens += mask.sum().item()
        # PAD对应mask为0，否则为1
        mask = mask * (dec_input != vocab[PAD]).float()
    return l / num_not_pad_tokens


def training(encoder, decoder, vocab, lr, batch_size, num_epochs):
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    val_bleu_list = []
    data_iter = Data.DataLoader(train_dataset, batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        l_sum = 0.0
        i = 0
        for X, Y in data_iter:
            i += 1
            X = X.to(device)
            Y = Y.to(device)
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            l = batch_loss(encoder, decoder, X, Y, vocab, loss)
            l.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            l_sum += l.item()
        val_bleu = evaluate(encoder, decoder, val_dataset, vocab, 1)
        val_bleu_list.append(val_bleu)
        print("Epoch %d, Train_loss: %.3f, Valid bleu: %.3f" %
              (epoch + 1, l_sum / i, val_bleu))
        if epoch == num_epochs-1:
            evaluate(encoder, decoder, test_dataset, vocab, 1, True)
            # torch.save(encoder.state_dict(),
            #            "./save_model_para/rnn_encoder.model")
            # torch.save(decoder.state_dict(),
            #            "./save_model_para/rnn_decoder.model")
    show_bleu_result(val_bleu_list, 'Seq2Seq')


def evaluate(encoder, decoder, dataset, vocab, batch_size, is_test=False):
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=False)
    preds = []
    total_bleu = 0.0
    k = 0
    des = []
    with torch.no_grad():
        for X, Y in data_iter:
            k += 1
            X = X.to(device)
            Y = Y.to(device)
            batch_size = X.shape[0]
            init_hidden = encoder.get_init_hidden()
            enc_outputs, enc_state = encoder(X, init_hidden)
            dec_state = decoder.get_init_hidden(enc_state)
            dec_input = torch.tensor([vocab[BOS]] * batch_size).to(device)
            dec_words = []
            actuals = []
            origin_des = []
            for di in range(config['summary_max_len']):
                dec_output, dec_state = decoder(dec_input.unsqueeze(1), dec_state, enc_outputs)
                topv, topi = dec_output.topk(1)
                # print(dec_output.shape)
                dec_input = topi.detach()
                # print(dec_input.shape)
                if topi.item() == vocab[EOS] or topi.item() == vocab[PAD]:
                    break
                else:
                    dec_words.append(
                        str([k for (k, v) in vocab.items() if v == topi.item()][0]))
                
            preds.append(' '.join(dec_words))
            origin_des_idx = X.tolist()[0]
            for j in range(len(origin_des_idx)):
                if origin_des_idx[j] == vocab[EOS] or origin_des_idx[j] == vocab[PAD]:
                    break
                origin_des.append(
                    str([k for (k, v) in vocab.items() if v == origin_des_idx[j]][0]))
            des.append(' '.join(origin_des))
            if not is_test:
                actuals_idx = Y.tolist()[0]
                for j in range(len(actuals_idx)):
                    if actuals_idx[j] == vocab[EOS] or actuals_idx[j] == vocab[PAD]:
                        break
                    actuals.append(
                        str([k for (k, v) in vocab.items() if v == actuals_idx[j]][0]))
                # print('....')
                # print('actual: ',[[actuals]])
                # print('predict: ',[dec_words])
                i_bleu = corpus_bleu(
                    [[actuals]], [dec_words], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
                # print('i_bleu', i_bleu)
                total_bleu += i_bleu
        if is_test:
            final_df = pd.DataFrame({'description': des, 'diagnosis': preds})
            final_df = final_df.rename_axis('index')
            final_df.to_csv('../../预测文件/predictions.csv')
    return total_bleu / k


# %%
def show_bleu_result(list, title):
    plt.plot(list, label="bleu")
    plt.title(title)
    plt.xlabel("epoch")
    plt.xticks(range(config['epochs']), range(1, config['epochs']+1))
    plt.ylabel("bleu")
    plt.legend()
    plt.savefig('./'+title+'.jpg')
    plt.show()

# %%
# 一些配置
config = {'description_max_len': 500,
          'summary_max_len': 100,
          'train_batch_size': 4,
          'lr': 5e-4,
          'epochs': 10}

# %%
def tensor_ready(vocab, des, dia):
    descriptions_dataset = word2index(
        vocab, des, config['description_max_len'], type='descriptions')
    summaries_dataset = word2index(
        vocab, dia, config['summary_max_len'], type='summary')
    return TensorDataset(descriptions_dataset, summaries_dataset)


# %%
SEED = 1458
# 随机数种子seed确定时，模型的训练结果将始终保持一致
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
# 读入数据，用空格作为分隔符分割成字符数组
data = pd.read_csv("data/train.csv")
test_df = pd.read_csv('data/test.csv')
descriptions = [np.array(data['description'])[i].split(' ')
                for i in range(len(data['description']))]
summaries = [np.array(data['diagnosis'])[i].split(' ')
             for i in range(len(data['diagnosis']))]
train_descriptions, val_descriptions, train_summaries, val_summaries = train_test_split(
    descriptions, summaries, test_size=0.1, random_state=1458, shuffle=True)
test_descriptions = [np.array(test_df['description'])[i].split(
    ' ') for i in range(len(test_df['description']))]
test_summaries = [np.array(test_df['diagnosis'])[i].split(
    ' ') for i in range(len(test_df['diagnosis']))]

# 建立词典
vocab = build_vocab(train_descriptions, train_summaries)
train_dataset = tensor_ready(vocab, train_descriptions, train_summaries)
val_dataset = tensor_ready(vocab, val_descriptions, val_summaries)
test_dataset = tensor_ready(vocab, test_descriptions, test_summaries)
vocab_size = len(vocab)
e_size = 256
h_size = 256
# encoder使用单层双向gru
encoder = EncoderRNN(vocab_size, e_size, h_size, n_layers=1)
decoder = DecoderRNN(vocab_size, e_size, h_size, n_layers=1)
encoder.to(device)
decoder.to(device)
training(encoder, decoder, vocab,
         config['lr'], config['train_batch_size'], config['epochs'])


