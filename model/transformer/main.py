import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.transformer.transformer import Transformer
from myutils.dataset import dataset
import logging

from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction


class TransformerTrain():
    def __init__(self, args):
        self.SRC_VOCAB_SIZE = args["vocab_size"]
        self.TGT_VOCAB_SIZE = args["vocab_size"]
        self.D_MODEL = 64
        self.D_FNN = 64
        self.NUM_HEADS = 2
        self.NUM_ENCODER_LAYERS = 2
        self.NUM_DECODER_LAYERS = 2
        self.BATCH_SIZE = args["batch_size"]
        self.DEVICE = args["device"]
        self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 1301, 1300, 1302
        self.version = args["version"]
        logging.basicConfig(level=logging.INFO, filename=f"log/transformer_{self.version}.txt", filemode="a+", format="%(asctime)s - %(levelname)s - %(message)s")

        model = Transformer(self.NUM_ENCODER_LAYERS,
                        self.NUM_DECODER_LAYERS,
                        self.NUM_HEADS,
                        self.SRC_VOCAB_SIZE,
                        self.TGT_VOCAB_SIZE,
                        self.D_MODEL,
                        self.D_FNN)

        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self.model = model.to(self.DEVICE)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)   # 该值在计算损失时将被忽略
        self.optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    """
    计算给定批次的填充掩码
    batch每个序列可能在末尾包含一些填充元素,
    在等于pad的地方为false,不等于pad的地方为true
    升维度
    """
    def calc_padding_mask(self, batch: torch.Tensor):
        padding_mask = torch.unsqueeze(batch != self.PAD_IDX, dim=1)  # [batch_size, 1, seq_len+1]
        return padding_mask
    
    def train_epoch(self, model, optimizer, output_per_batch=None):
        model.to(self.DEVICE)
        model.train()
        train_dataset = dataset('train')
        train_dataloader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, collate_fn=train_dataset.batch_process)
        total_loss = 0
        total_batch = 0
        for batch_id, (src, tgt) in enumerate(train_dataloader):
            total_batch += 1
            optimizer.zero_grad()
            src = src.to(self.DEVICE)   # [batch_size, seq_len+1]  # +1是因为加了<BOS>
            tgt = tgt.to(self.DEVICE)   # [batch_size, seq_len]
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_padding_mask = self.calc_padding_mask(src)
            tgt_padding_mask = self.calc_padding_mask(tgt_input)
            scores = model(src_padding_mask, tgt_padding_mask, src, tgt_input)  # [batch_size, seq_len, tgt_vocab_size]
            num_candidates = scores.size()[-1]
            loss = self.loss_fn(scores.reshape(-1, num_candidates),
                        tgt_output.long().reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()
            if output_per_batch is not None and batch_id % output_per_batch == 0:
                logging.info(f"Batch {batch_id}: loss = {loss.item()} avg_loss = {total_loss / (total_batch)}")
            # break
        return total_loss / total_batch


    def evaluate(self, model: Transformer):
        model.eval()
        val_dataset = dataset('val')
        val_dataloader = DataLoader(val_dataset, batch_size=self.BATCH_SIZE, shuffle=True, collate_fn=val_dataset.batch_process)
        total_loss = 0
        total_batch = 0
        for src, tgt in val_dataloader:
            total_batch += 1
            self.optimizer.zero_grad()
            src = src.to(self.DEVICE)
            tgt = tgt.to(self.DEVICE)
            tgt_input = tgt[:, :-1]  # 删去最后一个词作为输入
            tgt_output = tgt[:, 1:]  # 删去第一个词作为输出
            src_padding_mask = self.calc_padding_mask(src)
            tgt_padding_mask = self.calc_padding_mask(tgt_input)
            scores = model(src_padding_mask, tgt_padding_mask, src, tgt_input)
            num_candidates = scores.size()[-1]
            loss = self.loss_fn(scores.reshape(-1, num_candidates),
                        tgt_output.long().reshape(-1))
            loss.backward()
            self.optimizer.step()
            total_loss = total_loss + loss.item()
            # 计算bleu
            pred = scores.argmax(dim=-1)
            bleu = self.get_bleu(pred, tgt_output)
            # break
        return bleu

    def train(self):
        total_epoch = 100
        best_val_bleu = -1
        for epoch in range(total_epoch):
            logging.info(f"Epoch {epoch + 1} / {total_epoch}:")
            avg_loss = self.train_epoch(self.model, self.optimizer, 200)
            self.optimizer.param_groups[0]["lr"] /= 10
            logging.info(f"Epoch {epoch + 1} done, avg_loss = {avg_loss}")
            val_bleu = self.evaluate(self.model)
            logging.info(f"Epoch {epoch + 1} done, val_bleu = {val_bleu}")
            if val_bleu > best_val_bleu:
                best_val_bleu = val_bleu
                torch.save(self.model.state_dict(), "res/transformer-best.pkl")
                logging.info("model saved")
            # break

    def get_bleu(self, pred, Y):
        chencherry = SmoothingFunction()
        bleu_score = 0
        for pre, y in zip(pred, Y):
            bleu_score += sentence_bleu([y[y!=self.PAD_IDX].squeeze().tolist()], pre.tolist(), smoothing_function=chencherry.method1)
        return bleu_score/len(pred)
    
    def test(self):
        logging.info("test start")
        self.model.eval()
        test_dataset = dataset('test')
        test_dataloader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=True, collate_fn=test_dataset.batch_process)
        total_loss = 0
        total_batch = 0
        for src, tgt in test_dataloader:
            total_batch += 1
            self.optimizer.zero_grad()
            src = src.to(self.DEVICE)
            tgt = tgt.to(self.DEVICE)
            tgt_input = tgt[:, :-1]  # 删去最后一个词作为输入
            tgt_output = tgt[:, 1:]  # 删去第一个词作为输出
            src_padding_mask = self.calc_padding_mask(src)
            tgt_padding_mask = self.calc_padding_mask(tgt_input)
            scores = self.model(src_padding_mask, tgt_padding_mask, src, tgt_input)
            num_candidates = scores.size()[-1]
            loss = self.loss_fn(scores.reshape(-1, num_candidates),
                        tgt_output.long().reshape(-1))
            loss.backward()
            self.optimizer.step()
            total_loss = total_loss + loss.item()
            # 计算bleu
            pred = scores.argmax(dim=-1)
            bleu = self.get_bleu(pred, tgt_output)
        logging.info(f"test done, bleu = {bleu}")
        return bleu

if __name__ == "__main__":
    trainer = TransformerTrain()
    trainer.train()
    trainer.test()