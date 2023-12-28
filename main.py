import argparse
import time
import yaml
from train_val_test import train, test
from fituning import fituning, fituning_test
import torchinfo

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

parser = argparse.ArgumentParser()
localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
parser.add_argument('--version', type=str, default=str(localtime))
for k, v in config.items():
    parser.add_argument('--'+k, type=type(v), default=v)

args = parser.parse_args()
args = vars(args)

def run(args):
    args['model'] = 'lstmatt'
    modelname = args['model'].lower()
    if modelname == 'lstm2lstm':
        from model.lstm2lstm import Seq2Seq
        model = Seq2Seq(args)
        train(model, args)   # train 和 val 
        test(model, args)
    elif modelname == 'gru2gru':
        from model.gruatt2 import Seq2Seq
        model = Seq2Seq(args)
        train(model, args)
        test(model, args)
    elif modelname == 'lstmatt':
        from model.lstmatt import Seq2Seq
        model = Seq2Seq(args)
        train(model, args)
        test(model, args)
    elif modelname == 't5-medical':
        from model.t5_medical import get_model
        model, tokenizer = get_model()
        fituning(model, tokenizer, args)
        fituning_test(model, tokenizer, args)
    elif modelname == 't5-base':
        from model.t5_base import get_model
        model, tokenizer = get_model()
        fituning(model, tokenizer, args)
        fituning_test(model, args)

 
            
           
if __name__ == '__main__':
    run(args)