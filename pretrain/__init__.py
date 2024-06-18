from .methods import *
from .preprocess import *
import logging
import json


class Method:
    def __init__(self, args):
        self.method = args.method
        self.args = args
        self.path = self.path()
        self.logger = config_logger(self.path + '.log')
        self.logger.info(args)
        self.logger.info("Output path: " + self.path)
        self.amr_dataset = self.get_amr_dataset()
        self.text_dataset = self.get_text_dataset()
        self.model_name = args.bert_model_name


    def path(self):
        path = self.args.save_path
        return path


    def get_amr_dataset(self):
        amr_dataset = get_dataset_sub('PretrainedDatasets/' + self.args.amr_dataset_file)
        self.logger.info("AMR Dataset is ok.")
        return amr_dataset
       

    def get_text_dataset(self):
        json_file_path = 'PretrainedDatasets/' + self.args.text_dataset_file
        text_dataset = []  
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    text_dataset.append(json.loads(line))
            self.logger.info("Text Dataset is ok.")
        except Exception as e:
            self.logger.error(f"Failed to load Text dataset: {e}")
            text_dataset = None
        return text_dataset


    def train(self):
        if self.method == 'RGSN':
            return RGSN(self.args.times, self.args, self.path, self.logger, self.amr_dataset, self.text_dataset, self.model_name)
        elif self.method == 'GSN':
            return GSN(self.args.times, self.args, self.path, self.logger, self.amr_dataset, self.text_dataset, self.model_name)
        else:
            raise NotImplementedError


def config_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fhandler = logging.FileHandler(log_path, mode='w')
    shandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    shandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.addHandler(shandler)
    return logger

