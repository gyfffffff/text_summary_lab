from transformers import  AutoModelForSeq2SeqLM, AutoTokenizer

def get_model():
    checkpoint = "model/t5_base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    return model, tokenizer