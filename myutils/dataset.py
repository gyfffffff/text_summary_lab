import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset

def save_data():
    path_train = 'data/train.csv'
    path_test = 'data/test.csv'
    _train_data = pd.read_csv(path_train)[['description', 'diagnosis']]
    test_data = pd.read_csv(path_test)[['description', 'diagnosis']]
    # 打乱并拆分train_data
    train_data = _train_data.sample(frac=1).reset_index(drop=True)
    train_data = _train_data[:int(len(_train_data)*0.9)].values
    val_data = _train_data[int(len(_train_data)*0.9):].values
    # 保存
    pickle.dump(train_data[:, 0], open('data/train_x.pkl', 'wb'))
    pickle.dump(train_data[:, 1], open('data/train_y.pkl', 'wb'))
    pickle.dump(val_data[:, 0], open('data/val_x.pkl', 'wb'))
    pickle.dump(val_data[:, 1], open('data/val_y.pkl', 'wb'))
    pickle.dump(test_data['description'].values, open('data/test_x.pkl', 'wb'))
    pickle.dump(test_data['diagnosis'].values, open('data/test_y.pkl', 'wb'))


def get_vocab_size():
    train_data = pickle.load(open('data/train_x.pkl', 'rb'))
    vocab = set()
    for data in train_data:
        vocab.update(data.split(' '))
    vocab = list(vocab)
    vocab = [int(v) for v in vocab]
    print(max(vocab))

class dataset(Dataset):
    def __init__(self, phase):
        self.x = pickle.load(open('data/'+phase+'_x.pkl', 'rb'))
        self.y = pickle.load(open('data/'+phase+'_y.pkl', 'rb'))
         
    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        assert len(self.x) == len(self.y)
        return self.x[idx], self.y[idx]
    
    def batch_process(self, batch):
        # 填充
        x, y, x_len, y_len = [], [], [], []
        for _x, _y in batch:
            x_len.append(len(_x))
            y_len.append(len(_y))
            x.append(_x)
            y.append(_y)
        max_x_len = max(x_len) if max(x_len) < 200 else 200
        max_y_len = max(y_len) if max(y_len) < 100 else 100

        # <BOS> 1292, <EOS> 1293, <PAD> 1294
        BOS = '1300'
        PAD = '1301'
        EOS = '1302'
        x = [_x.split(' ') + [PAD]*(max_x_len - len(_x.split(' '))) + [EOS] for _x in x]
        y = [[BOS] + _y.split(' ') + [PAD]*(max_y_len - len(_y.split(' '))) + [EOS] for _y in y]

        x = [[int(__x) for __x in _x] for _x in x]
        y = [[int(__y) for __y in _y] for _y in y]

        # 转换为tensor
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y
    
    def val_process(self, batch):
        x, y = [], []
        for _x, _y in batch:
            x.append(_x)
            y.append(_y)
        x = [[int(__x) for __x in _x.split(" ")] for _x in x]
        y = [[int(__y) for __y in _y.split(" ")] for _y in y]
        # 转换为tensor
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y
    
class fituning_dataset(Dataset):
    def __init__(self, phase, tokenizer):
        self.tokenizer = tokenizer
        self.description_len = 500
        self.diagnosis_len = 100
        if phase == 'train':
            df = pd.read_csv('data/train.csv')[['description', 'diagnosis']]
            df.description = 'summarize: ' + df.description
            train_dataset = df.sample(frac=0.9, random_state=1717).reset_index(drop=True)
            self.description = train_dataset.description
            self.diagnosis = train_dataset.diagnosis
        elif phase == 'val':
            df = pd.read_csv('data/train.csv')[['description', 'diagnosis']]
            df.description = 'summarize: ' + df.description
            train_dataset = df.sample(frac=0.9, random_state=1717).reset_index(drop=True) 
            val_dataset = df.drop(train_dataset.index).reset_index(drop=True)  
            self.description = val_dataset.description
            self.diagnosis = val_dataset.diagnosis         
        elif phase == 'test':
            test_df = pd.read_csv('data/test.csv')[['description', 'diagnosis']]
            test_df.description = 'summarize: ' + test_df.description        
            test_dataset = test_df.reset_index(drop=True)
            self.description = test_dataset.description
            self.diagnosis = test_dataset.diagnosis


    def __len__(self):
        return len(self.description)

    def __getitem__(self, index):
        diagnosis = str(self.diagnosis[index])
        description = str(self.description[index])
        source = self.tokenizer(
            [description], truncation=True, padding='max_length', max_length=self.description_len, return_tensors='pt')
        target = self.tokenizer(
            [diagnosis], truncation=True, padding='max_length', max_length=self.diagnosis_len, return_tensors='pt')
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

if __name__ == '__main__':
    # save_data()
    get_vocab_size()
