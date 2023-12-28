# %%
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging
logging.basicConfig(format='%(asctime)s %(message)s',filename="t5-base.log",
        filemode='a+',
        level=logging.INFO)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


# %%
class NewDataset(Dataset):
    def __init__(self, data, tokenizer, des_len, dia_len):
        self.tokenizer = tokenizer
        self.description = data.description
        self.diagnosis = data.diagnosis
        self.des_len = des_len
        self.dia_len = dia_len

    def __len__(self):
        return len(self.description)

    def __getitem__(self, index):
        diagnosis = str(self.diagnosis[index])
        description = str(self.description[index])
        source = self.tokenizer.batch_encode_plus(
            [description], truncation=True, padding='max_length', max_length=self.des_len,  return_tensors='pt')
        target = self.tokenizer.batch_encode_plus(
            [diagnosis], truncation=True, padding='max_length', max_length=self.dia_len, return_tensors='pt')
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_mask': target_mask.to(dtype=torch.long)
        }

# %%
def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    loss_sum = 0.0
    i = 0
    for _, data in enumerate(loader, 0):
        i += 1
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        truth_labels = y[:, 1:].clone().detach()
        # 所有被设置为-100的label在计算loss时会被忽略
        truth_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        outputs = model(input_ids=ids, attention_mask=mask,
                        decoder_input_ids=y_ids, labels=truth_labels)
        # 当设置了labels属性时，会计算交叉熵损失
        loss = outputs[0]
        loss_sum += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # path = './t5_model.pth'
        # state = {'model': model.state_dict(
        # ), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        # torch.save(state, path)
    return loss_sum / i


# %%
def validate(tokenizer, model, device, loader, is_test=False):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)
            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=SUMMARY_MAX_LEN,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            if not is_test:
                act = [tokenizer.decode(
                    item, skip_special_tokens=True, clean_up_tokenization_spaces=True) for item in y]
                actuals.extend(act)
            preds = [tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            predictions.extend(preds)
    return predictions, actuals



# %%
SEED = 1458
MAX_LEN = 500
SUMMARY_MAX_LEN = 100


def main():
    num_epochs = 155
    lr = 2e-4
    train_params = {
        'batch_size': 8,
        'shuffle': True
    }

    val_params = {
        'batch_size': 8,
        'shuffle': False
    }

    test_params = {
        'batch_size': 8,
        'shuffle': False
    }
    # 随机数种子seed确定时，模型的训练结果将始终保持一致
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    df = pd.read_csv('../data/train.csv')
    df = df[['description', 'diagnosis']]
    df.description = 'summarize: ' + df.description
    test_df = pd.read_csv('../data/test.csv')
    test_df = test_df[['description', 'diagnosis']]
    origin_test_df = test_df.copy(deep=True)
    test_df.description = 'summarize: ' + test_df.description

    train_size = 0.9
    train_dataset = df.sample(
        frac=train_size, random_state=SEED).reset_index(drop=True)
    val_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    test_dataset = test_df.reset_index(drop=True)

    train_set = NewDataset(train_dataset, tokenizer, MAX_LEN, SUMMARY_MAX_LEN)
    val_set = NewDataset(val_dataset, tokenizer, MAX_LEN, SUMMARY_MAX_LEN)
    test_set = NewDataset(test_dataset, tokenizer, MAX_LEN, SUMMARY_MAX_LEN)

    training_loader = DataLoader(train_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    test_loader = DataLoader(test_set, **test_params)

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    
    best_bleu = -1
    for epoch in range(1, num_epochs+1):
        loss = train(epoch, tokenizer, model, device,
                     training_loader, optimizer)
        predictions, autuals = validate(tokenizer, model, device, val_loader)
        autuals = [[x.split(' ')] for x in autuals]
        predictions = [x.split(' ') for x in predictions]
        bleu = corpus_bleu(autuals, predictions,
                           weights=(0.25, 0.25, 0.25, 0.25))
        print("Epoch %d, Train_loss: %.3f, Valid bleu: %.3f" %
              (epoch, loss, bleu))
        logging.info(f"epoch {epoch}, train-loss {loss}, val-bleu {bleu}")
        if bleu > best_bleu:
            torch.save(model.state_dict(), open('t5-base-best.pkl', 'wb'))
    # path = './model.pth'
    # checkpoint = torch.load(path)
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # epoch = checkpoint(['epoch'])
    test_predictions, _ = validate(
        tokenizer, model, device, test_loader, True)
    # print(predictions)
    final_df = pd.DataFrame(
        {'description': origin_test_df['description'], 'diagnosis': test_predictions})
    final_df = final_df.rename_axis('index')
    final_df.to_csv('t5-base-predictions.csv')


if __name__ == '__main__':
    main()


