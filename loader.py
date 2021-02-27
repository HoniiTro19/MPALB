import json
import ipdb
import torch
import argparse
from path import Path
from configure import ConfigParser
from torch.utils.data import TensorDataset, IterableDataset, DataLoader

config = ConfigParser()
path = Path(config)
use_gpu = config.getboolean("train", "use_gpu")
device_num = config.getint("train", "gpu_device")
cuda = torch.device("cuda:%d" % device_num)
cpu = torch.device("cpu")
device = cuda if use_gpu else cpu


def SmallDataset(json_path):
    file = open(json_path, "r", encoding="UTF-8")
    lines = json.loads(file.read())
    file.close()
    fact, doc_len, accusation, article, imprison = list(), list(), list(), list(), list()
    for line in lines:
        fact.append(line["fact"])
        doc_len.append(line["len"])
        accusation.append(line["accusation"])
        article.append(line["article"])
        imprison.append(line["imprison"])

    fact = torch.tensor(fact, device=device, dtype=torch.int64)
    doc_len = torch.tensor(doc_len, device=device, dtype=torch.int64)
    accusation = torch.tensor(accusation, device=device, dtype=torch.int64)
    article = torch.tensor(article, device=device, dtype=torch.int64)
    imprison = torch.tensor(imprison, device=device, dtype=torch.int64)
    data = TensorDataset(fact, doc_len, accusation, article, imprison)
    return data

class LargeDataset(IterableDataset):
    def __init__(self, set_type):
        super(LargeDataset).__init__()
        preprocess_path = path.preprocess_path
        split_train_num = config.getint("preprocess", "split_train_num")
        split_test_num = config.getint("preprocess", "split_test_num")
        if set_type == "train":
            self.data_list = [preprocess_path + "train_large_w2v-%d.json" % idx for idx in range(split_train_num)]
        elif set_type == "test":
            self.data_list = [preprocess_path + "test_large_w2v-%d.json" % idx for idx in range(split_test_num)]
        else:
            raise Exception("Cannot recognize the set type %s, suppose to be 'train/test'" % set_type)

    def reload_file(self, file_name):
        with open(file_name, "r", encoding="UTF-8") as file:
            self.lines = json.loads(file.read())

    def get_data(self):
        for file_name in self.data_list:
            self.reload_file(file_name)
            for line in self.lines:
                fact = torch.tensor(line["fact"], device=device, dtype=torch.int64)
                doc_len = torch.tensor(line["len"], device=device, dtype=torch.int64)
                accusation = torch.tensor(line["accusation"], device=device, dtype=torch.int64)
                article = torch.tensor(line["article"], device=device, dtype=torch.int64)
                imprison = torch.tensor(line["imprison"], device=device, dtype=torch.int64)
                yield fact, doc_len, accusation, article, imprison

    def __iter__(self):
        return self.get_data()

if __name__ == "__main__":
    path = Path(config)
    small_trainset = dataset_loader(path.train_small_w2v_path)