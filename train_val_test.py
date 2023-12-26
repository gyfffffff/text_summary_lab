from utils.logger import Logger
from utils.dataset import dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import os
import time
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction

from rouge_chinese import Rouge



def train(model, args):
    log = Logger(args)

    epoches = args['epoches']
    lr = args['lr']
    lr_decay = args['lr_decay']
    optim = args['optim']
    patience = args['patience']
    batch_size = args['batch_size']
    device = args['device']
    hidden_size = args['hidden_size']
    emb_size = args['embed_size']
    num_layers = args['num_layers']
    vocab_size = args['vocab_size']
    log_dir = args['log_dir']
    res_dir = args['res_dir']
    data_dir = args['data_dir']
    version = args['version']
    clip = args['clip']
    max_norm = args['max_norm']

    model.to(device)
    train_dataset = dataset('train')
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.batch_process)
    val_dataset = dataset('val')
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=val_dataset.batch_process)

    if optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # lossfun = nn.CrossEntropyLoss()
    
    log.write_config()   # 打印配置信息，说明训练开始
    log.write('trainable params: '+ str(get_param_num(model)))
    log.write([p.numel() for p in model.parameters() if p.requires_grad])
    # train_acc_history = []
    train_loss_history = []
    val_blue_history = []
    val_rouge1_history = []
    val_rougeL_history = []
    # val_loss_history = []
    best_val_blue = -1
    no_improve = 0  # 验证集效果没提升的轮数

    start = time.time()
    for epoch in range(1, epoches+1):
        model.train()
        train_loss = 0
        for i, (X, Y) in enumerate(trainloader):
            X, Y = X.to(device), Y.to(device)
            loss = model(X, Y)   # forward
            train_loss += loss
            loss.backward()   # backward
            if clip:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
            # break
            if i%200 == 0:
                log.write('====> epoch: {}/{} batch: {} loss: {}'.format(epoch, epoches, i, loss.item()))
        train_loss /= len(trainloader)
        train_loss_history.append(train_loss.item())

        model.eval() 
        val_blue, val_rouge1, val_rougeL = 0, 0, 0
        for i, (X_val, Y_val) in enumerate(valloader):
            X_val, Y_val = X_val.to(device), Y_val.to(device)
            val_pred = model.summary(X_val)
            val_blue += get_BLUE(val_pred, Y_val)  # 这个batch的均值
            _val_rouge1, _val_rougeL = get_rouge(val_pred, Y_val)  # 这个batch的均值
            val_rouge1 += _val_rouge1
            val_rougeL += _val_rougeL
            # break
        val_blue /= len(valloader)
        val_rouge1 /= len(valloader)
        val_rougeL /= len(valloader)
        val_blue_history.append(val_blue)
        val_rouge1_history.append(val_rouge1)
        val_rougeL_history.append(val_rougeL)
        log.write('[epoch: {}/{}] train-loss: {} val-blue: {} val-rouge1: {} val-rougeL: {}'.format(epoch, epoches, train_loss, val_blue, val_rouge1, val_rougeL))
                
        if val_blue > best_val_blue:   # 是否保存最好的模型
            best_val_blue = val_blue
            torch.save(model.state_dict(), open(os.path.join(res_dir,f'{version}_best_model.pth'), 'wb'))
            best_epoch = epoch
            log.write('best model saved')
            log.write('best val blue up to now: {}'.format(best_val_blue))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve == 2 and lr_decay:
                 lr /= lr_decay
                 log.write('lr decay. ')
            log.write(f'patience: {no_improve}/{patience}')
        if no_improve >= patience:
            log.write('early stopping\n')
            break

    time_consum = time.time() - start
    log.write('train time consumption: {:.2f}s\n'.format(time_consum))   
    pickle.dump({'train_loss_history': train_loss_history,
                    'val_rouge1_history': val_rouge1_history,
                    'val_rougeL_history': val_rougeL_history,
                    'val_blue_history': val_blue_history},
                    open(os.path.join(res_dir,f'{version}_history.pkl'), 'wb'))
    plot(version, best_epoch)
    log.write('history plot saved at res/{}_history.png'.format(version))

def test(model, args):
    log = Logger(args)
    device = args['device']
    batch_size = args['batch_size']
    version = args['version']
    log.write(f'\n========> test: {version} <========')
    test_dataset = dataset('test')
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=test_dataset.batch_process)
    model.load_state_dict(torch.load(os.path.join(args['res_dir'], f'{version}_best_model.pth')))
    model.to(device)
    model.eval()
    start = time.time()
    test_blue, test_rouge1, test_rougeL = 0, 0, 0
    for i, (X, Y) in enumerate(testloader):
        X, Y = X.to(device), Y.to(device)
        pred = model.summary(X)
        test_blue += get_BLUE(pred, Y)
        _test_rouge1, _test_rougeL = get_rouge(pred, Y)  # 这个batch的均值
        test_rouge1 += _test_rouge1
        test_rougeL += _test_rougeL

    time_consum = time.time() - start
    log.write('test time consumption: {:.2f}s\n'.format(time_consum))
    test_blue /= len(testloader)
    test_rouge1 /= len(testloader)
    test_rougeL /= len(testloader)
    log.write('test blue: {}  test rouge-1: {} test rouge-L: {}'.format(test_blue, test_rouge1, test_rougeL))

def plot(version, best_epoch):
    data = pickle.load(open(f'res/{version}_history.pkl', 'rb'))

    # 绘制历史曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4))
    # plt.plot(data['train_loss_history'], label='train loss', linestyle='--', c='green')
    plt.plot(data['val_rouge1_history'], label='val rouge1', c='red')
    plt.plot(data['val_rougeL_history'], label='val rougeL', c='orange')
    plt.plot(data['val_blue_history'], label='val blue', c='blue')
    plt.plot([best_epoch-1, best_epoch-1], [0, 1], linestyle='--', color='brown', label='best epoch')
    plt.tight_layout()
    plt.legend()
    plt.xlabel('epoches')
    plt.savefig(f'res/{version}_history.png')

    plt.figure(figsize=(4, 4))
    plt.plot(data['train_loss_history'], label='train loss', linestyle='--', c='green')
    plt.legend()
    plt.xlabel('epoches')
    plt.savefig(f'res/{version}_history.png')


def get_BLUE(pred, Y):
    chencherry = SmoothingFunction()
    bleu_score = 0
    for pre in pred:
        bleu_score += sentence_bleu(Y.squeeze().tolist(), pre, smoothing_function=chencherry.method1)
    return bleu_score/len(pred)

def get_rouge(pred, Y):
    rouge = Rouge()
    pred = [' '.join([str(p) for p in pred])]  # pred: [['14', '30', '66', '75', '80', '33', '19', '35', '12', ...]]
    Y = [' '.join([str(y) for y in Y.squeeze().tolist()])]  # Y: [['1300', '822', '108', '28', '107', '104', '113', '110', '15', ...]]
    try:
        rouge_score = rouge.get_scores(pred, Y, avg=True)
    except ValueError:
        rouge_score = {'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0}}
    return rouge_score["rouge-1"]["f"], rouge_score["rouge-l"]["f"]

def get_param_num(model):
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_num