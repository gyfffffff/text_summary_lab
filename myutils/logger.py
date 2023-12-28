import logging
import os

class Logger:
    def __init__(self, args):
        self.args = args
        log_dir = args['log_dir']
        version = args['model']+'_'+args['version']
        logging.basicConfig(format='%(asctime)s %(message)s',
        filename=os.path.join(log_dir, version+'.txt'),
        filemode='a+',
        level=logging.INFO)
    def write_config(self):
        for k, v in self.args.items():
            logging.info(k+'\t'+str(v))
    def write(self, text):
        logging.info(text)