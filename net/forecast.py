import ipdb
import json
import torch
from path import Path
import torch.nn as nn
import torch.nn.functional as F
from utils import load_list_dict
from configure import ConfigParser

class Forecast(nn.Module):
    def __init__(self, task, hidden_size):
        super(Forecast, self).__init__()
        config = ConfigParser()
        path = Path(config)
        self.task = task
        is_remove_unfrequent = config.getboolean("preprocess", "is_remove_unfrequent")
        train_set = config.get("train", "train_set")
        if is_remove_unfrequent:
            accu_path = path.meta_accu_frequent_small_path if train_set == "small" else path.meta_accu_frequent_large_path
            law_path = path.meta_law_frequent_small_path if train_set == "small" else path.meta_law_frequent_large_path
        else:
            accu_path = path.meta_accu_path
            law_path = path.meta_law_path
        imprison_path = path.meta_imprison_path

        if task == "accu":
            self.accu, _ = load_list_dict(accu_path)
            tar_num = len(self.accu)
        elif task == "law":
            self.law, _ = load_list_dict(law_path)
            tar_num = len(self.law)
        elif task == "imprison":
            self.imprison, _ = load_list_dict(imprison_path)
            tar_num = len(self.imprison)
        else:
            raise Exception("Cannot recognize the task %s, suppose to be 'accu', 'law' or 'imprison'" % task)
        self.linear_forecast = nn.Linear(hidden_size, tar_num, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden):
        '''
        :param hidden: [batch_size, hidden]
        :return: [batch_size, seq_len] or [batch_size]
        '''
        outputs = self.linear_forecast(hidden)
        tags = torch.argmax(self.sigmoid(outputs), dim=1).tolist()
        return outputs, tags
