import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from transformer import Transformer
import pickle
import os
os.chdir("model/transformer")

# dcws = de_core_web_sm.load()

"""
Initializations and Definitions
"""

multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
torch.manual_seed(0)

SRC_LANGUAGE = "de"
TGT_LANGUAGE = "en"
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

token_transform = {}
vocab_transform = {}
text_transform = {}


def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


def sequential_transforms(*transforms):
    def func(text):
        for transform in transforms:
            text = transform(text)
        return text
    return func


def tensor_transform(token_ids: List):
    return torch.cat([torch.Tensor([BOS_IDX]),
                      torch.Tensor(token_ids),
                      torch.Tensor([EOS_IDX])])


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip('\n')))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip('\n')))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


if not os.path.exists(r".\token_transforms.pkl"):
    token_transform[SRC_LANGUAGE] = get_tokenizer("spacy",
                                                  language="de_core_news_sm")
    token_transform[TGT_LANGUAGE] = get_tokenizer("spacy",
                                                  language="en_core_web_sm")
    train_iter = Multi30k(split="train",
                          language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    vocab_transform[SRC_LANGUAGE] = build_vocab_from_iterator(
        yield_tokens(train_iter, SRC_LANGUAGE),
        min_freq=1,
        specials=special_symbols,
        special_first=True
    )
    vocab_transform[TGT_LANGUAGE] = build_vocab_from_iterator(
        yield_tokens(train_iter, TGT_LANGUAGE),
        min_freq=1,
        specials=special_symbols,
        special_first=True
    )
    vocab_transform[SRC_LANGUAGE].set_default_index(UNK_IDX)
    vocab_transform[TGT_LANGUAGE].set_default_index(UNK_IDX)
    with open(r".\token_transforms.pkl", "wb") as f:
        pickle.dump((token_transform, vocab_transform), f)
else:
    with open(r".\token_transforms.pkl", "rb") as f:
        token_transform, vocab_transform = pickle.load(f)


for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # So that the words not contained in the dictionary will be indexed as UNK
    text_transform[ln] = sequential_transforms(token_transform[ln],
                                               vocab_transform[ln],
                                               tensor_transform)


def calc_padding_mask(batch: torch.Tensor):
    padding_mask = torch.unsqueeze(batch != PAD_IDX, dim=1)
    return padding_mask


"""
Hyper-parameters
"""

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
D_MODEL = 512
D_FNN = 512
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
BATCH_SIZE = 128
DEVICE = "cuda"


"""
Training
"""

print("Begin training.")

model = Transformer(NUM_ENCODER_LAYERS,
                    NUM_DECODER_LAYERS,
                    NUM_HEADS,
                    SRC_VOCAB_SIZE,
                    TGT_VOCAB_SIZE,
                    D_MODEL,
                    D_FNN)

for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)

model = model.to(DEVICE)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


def train_epoch(model, optimizer, output_per_batch=None):
    model.to(DEVICE)
    model.train()
    train_iter = Multi30k(split="train",
                          language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE,
                                  collate_fn=collate_fn)
    total_loss = 0
    total_batch = 0
    for batch_id, (src, tgt) in enumerate(train_dataloader):
        total_batch += 1
        optimizer.zero_grad()
        src = src.T.to(DEVICE)
        tgt = tgt.T.to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        src_padding_mask = calc_padding_mask(src)
        tgt_padding_mask = calc_padding_mask(tgt_input)
        scores = model(src_padding_mask, tgt_padding_mask, src, tgt_input)
        num_candidates = scores.size()[-1]
        loss = loss_fn(scores.reshape(-1, num_candidates),
                       tgt_output.long().reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        if output_per_batch is not None and batch_id % output_per_batch == 0:
            print(f"Batch {batch_id}: loss = {loss.item()}"
                  f" avg_loss = {total_loss / (total_batch)}")
    return total_loss / total_batch


def evaluate(model: Transformer):
    model.eval()
    val_iter = Multi30k(split="valid",
                        language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE,
                                collate_fn=collate_fn)
    total_loss = 0
    total_batch = 0
    for src, tgt in val_dataloader:
        total_batch += 1
        optimizer.zero_grad()
        src = src.T.to(DEVICE)
        tgt = tgt.T.to(DEVICE)
        tgt_input = tgt[:, :-1]  # 删去最后一个词作为输入
        tgt_output = tgt[:, 1:]  # 删去第一个词作为输出
        src_padding_mask = calc_padding_mask(src)
        tgt_padding_mask = calc_padding_mask(tgt_input)
        scores = model(src_padding_mask, tgt_padding_mask, src, tgt_input)
        num_candidates = scores.size()[-1]
        loss = loss_fn(scores.reshape(-1, num_candidates),
                       tgt_output.long().reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
    return total_loss / total_batch


total_epoch = 10
for epoch in range(total_epoch):
    print(f"Epoch {epoch + 1} / {total_epoch}:")
    avg_loss = train_epoch(model, optimizer, 1)
    optimizer.param_groups[0]["lr"] /= 10
    print(f"Epoch {epoch + 1} done, avg_loss = {avg_loss}")
