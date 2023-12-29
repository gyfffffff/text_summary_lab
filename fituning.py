import logging
from myutils.logger import Logger
from myutils.dataset import fituning_dataset
import os
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def fituning(model, tokenizer, args):
    log = Logger(args)

    epoches = args['epoches']
    lr = args['lr']
    optim = args['optim']
    batch_size = args['batch_size']
    device = args['device']
    version = args['version']
    res_dir = args['res_dir']

    
    traindataset = fituning_dataset('train', tokenizer)
    valdataset = fituning_dataset('val', tokenizer)
    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)
    
    if optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_history = []
    model.to(device)
    for epoch in range(1, epoches+1):
        model.train()
        train_loss = 0
        
        for i, data in enumerate(trainloader):
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)
            y = data['target_ids'].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            truth_labels = y[:, 1:].clone().detach()
            truth_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            outputs = model(input_ids=ids, attention_mask=mask,
                            decoder_input_ids=y_ids, labels=truth_labels)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i%400==0:
                logging.info(f'===> batch {i}, loss: {loss.item()}')
            # break
        train_loss /= len(trainloader)
        train_loss_history.append(train_loss)

        model.eval()
        Ys, predictions = [],[]
        best_bleu = -1
        for _, data in enumerate(valloader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)
            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=100,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            act = [tokenizer.decode(
                item, skip_special_tokens=True, clean_up_tokenization_spaces=True) for item in y]
            Ys.extend(act)
            preds = [tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            predictions.extend(preds)
            # break
        bleu = get_bleu(Ys, predictions)
        log.write('Epoch: {}, Train_loss: {}, Valid bleu: {}'.format(epoch, train_loss, bleu))
        if bleu > best_bleu:
            torch.save(model.state_dict(), open(os.path.join(res_dir, f'{version}_best_model.pth'), 'wb'))
            best_bleu = bleu
        
        # break 

def fituning_test(model, tokenizer, args):
    log = Logger(args)
    device = args['device']
    version = args['version']
    batch_size = args['batch_size']
    log.write(f'\n========> test: {version} <========')
    testdataset = fituning_dataset('test', tokenizer)
    testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True)
    model.load_state_dict(torch.load(os.path.join(args['res_dir'], f'{version}_best_model.pth')))
    model.to(device)
    model.eval()
    Ys, predictions = [], []
    test_blue = 0
    for i, data in enumerate(testloader, 0):
        y = data['target_ids'].to(device, dtype=torch.long)
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)
        generated_ids = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_length=100,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        act = [tokenizer.decode(
                item, skip_special_tokens=True, clean_up_tokenization_spaces=True) for item in y]
        Ys.extend(act)
        preds = [tokenizer.decode(
            g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        predictions.extend(preds)    
        test_blue += get_bleu(Ys, predictions)
    test_blue /= len(testloader)
    log.write('Test bleu: {}'.format(test_blue))



def get_bleu(Y, pred):
    Y = [x.split(' ') for x in Y]
    pred = [y.split(' ') for y in pred]
    # bleu = sentence_bleu(Ys, predictions, weights=(0.25, 0.25, 0.25, 0.25))
    # return bleu
    chencherry = SmoothingFunction()
    bleu_score = 0
    for (pre, y) in zip(pred, Y):
        bleu_score += sentence_bleu([y], pre, smoothing_function=chencherry.method1)
    return bleu_score/len(pred)