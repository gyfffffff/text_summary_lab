from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
ds = load_dataset('csv', data_files='data/train.csv')['train']

splited_ds = ds.train_test_split(test_size=0.1, seed=42)

train_ds = splited_ds['train']
val_ds = splited_ds['test']

tokenizer = AutoTokenizer.from_pretrained("Langboat/mengzi-t5-base")

def process_func(examples):
    content = ['摘要生成：\n' + e for e in examples['description']]
    inputs = tokenizer(content, truncation=True, max_length=218)
    lables = tokenizer(text_target=examples['diagnosis'], truncation=True, max_length=32)
    inputs['labels'] = lables['input_ids']
    return inputs

train_ds = train_ds.map(process_func, batched=True)
val_ds = val_ds.map(process_func, batched=True)
# print(tokenizer.decode(tokenized_ds[0]['input_ids']))
# print(tokenizer.decode(tokenized_ds[0]['labels']))

model = AutoModelForSeq2SeqLM.from_pretrained("Langboat/mengzi-t5-base")

import numpy as np
from rouge_chinese import Rouge

rouge = Rouge()

def compute_metric(evalPred):
    predictions, labels = evalPred
    decode_pred = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decode_label = tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge_score = rouge.get_scores(decode_pred, decode_label, avg=True)
    return {
        "rouge1": rouge_score["rouge-1"]["f"],
        "rouge2": rouge_score["rouge-2"]["f"],
        "rougeL": rouge_score["rouge-l"]["f"],
    }

# 配置训练参数
args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=8,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    metric_for_best_model="rouge1",
    predict_with_generate=True,   # 否则无法做评估
)

# 创建训练器
trainer = Seq2SeqTrainer(
    args = args,
    model = model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metric,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer)
)

# 开始训练
trainer.train()