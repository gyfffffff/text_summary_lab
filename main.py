import argparse
import time
import yaml
from train_val_test import train, test
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
    modelname = args['model'].lower()
    # modelname = 'gru2gru'
    if modelname == 'lstm2lstm':
        from model.lstm2lstm import Seq2Seq
        model = Seq2Seq(args)
    elif modelname == 'gru2gru':
        from model.gruatt2 import Seq2Seq
        model = Seq2Seq(args)
    # print(torchinfo.summary(model, input_size=(1, 28, 28), batch_dim=0))
    train(model, args)
    test(model, args)

if __name__ == '__main__':
    run(args)