import os
import ipdb
import json
import torch
import numpy as np
from tqdm import tqdm
from path import Path
from demo import Demo
import torch.nn as nn
from eval import Eval
import torch.optim as optim
from configure import ConfigParser
from net.full_model import FullModel
from preprocessor import Preprocessor
from torch.utils.data import DataLoader
from utils import load_list_dict, FocalLoss
from loader import SmallDataset, LargeDataset

class Train():
    def __init__(self):
        self.config = ConfigParser()
        self.path = Path(self.config)
        # Demo
        is_statistics = self.config.getboolean("demo", "is_statistics")
        self.demo = Demo()
        if is_statistics:
            self.demo.statistics()

        # Preprocess
        is_preprocess = self.config.getboolean("preprocess", "is_preprocess")
        preprocessor = Preprocessor()
        if is_preprocess:
            preprocessor.preprocess()

        # Cuda
        use_gpu = self.config.getboolean("train", "use_gpu")
        device_num = self.config.getint("train", "gpu_device")
        cuda = torch.device("cuda:%d" % device_num)
        cpu =  torch.device("cpu")
        if use_gpu:
            if not torch.cuda.is_available():
                raise Exception("Cuda device is not available")
            self.device = cuda
        else:
            self.device = cpu

        # Accu, Law, Imprison
        is_remove_unfrequent = self.config.getboolean("preprocess", "is_remove_unfrequent")
        self.train_set = self.config.get("train", "train_set")
        if is_remove_unfrequent:
            accu_path = self.path.meta_accu_frequent_small_path if self.train_set == "small" else self.path.meta_accu_frequent_large_path
            law_path = self.path.meta_law_frequent_small_path if self.train_set == "small" else self.path.meta_law_frequent_large_path
        else:
            accu_path = self.path.meta_accu_path
            law_path = self.path.meta_law_path
        imprison_path = self.path.meta_imprison_path
        self.list_accu, _ = load_list_dict(accu_path)
        self.list_law, _ = load_list_dict(law_path)
        self.list_imprison, _ = load_list_dict(imprison_path)

    def train_file(self):
        is_pretrain = self.config.getboolean("train", "is_pretrain")
        model_path = self.path.model_path
        model_pretrain_path = model_path + "model-%d.pkl" % self.config.getint("train", "last_model")
        model = FullModel()
        if is_pretrain:
            if os.path.exists(model_pretrain_path):
                model.load_state_dict(torch.load(model_pretrain_path))
            else:
                raise Exception("No pretrained model accessible.")
        model.to(self.device)

        batch_size = self.config.getint("train", "batch_size")
        is_shuffle = self.config.getboolean("train", "is_shuffle")
        if self.train_set == "small":
            train_small_w2v_path = self.path.train_small_w2v_path
            valid_small_w2v_path = self.path.valid_small_w2v_path
            train_dataset = SmallDataset(train_small_w2v_path)
            valid_dataset = SmallDataset(valid_small_w2v_path)
            train_data = DataLoader(train_dataset, batch_size, is_shuffle, drop_last=True)
            valid_data = DataLoader(valid_dataset, batch_size, is_shuffle, drop_last=True)
        elif self.train_set == "large":
            train_dataset = LargeDataset("train")
            train_data = DataLoader(train_dataset, batch_size, drop_last=True)
        else:
            raise Exception("Cannot recognize the dataset type %s, suppose to be 'small' or 'large'" % self.train_set)
        epochs = self.config.getint("train", "epoch")
        lr = self.config.getfloat("train", "lr")
        embed_size = self.config.getint("preprocess", "embed_size")
        hidden_size = self.config.getint("train", "hidden_size")
        accu_hidden_size = self.config.getint("train", "accu_hidden_size")
        optimizer = optim.Adam(model.parameters(), lr=lr)

        criterion_accu = FocalLoss(class_num=len(self.list_accu)).to(self.device)
        criterion_article = FocalLoss(class_num=len(self.list_law)).to(self.device)
        criterion_imprison = FocalLoss(class_num=len(self.list_imprison)).to(self.device)
        epoch_list = tqdm(range(epochs))

        best_f1 = 0
        best_epoch = 0
        train_losses, valid_losses = list(), list()
        valid_accu_f1, valid_article_f1, valid_imprison_f1 = list(), list(), list()
        for epoch in epoch_list:
            # Train
            model.train()
            eval_train_accu = Eval(len(self.list_accu))
            eval_train_article = Eval(len(self.list_law))
            eval_train_imprison = Eval(len(self.list_imprison))
            total_train_loss = 0
            for data in train_data:
                model.init_hidden()
                optimizer.zero_grad()
                outputs_accu, outputs_article, outputs_imprison, tags_accu, tags_article, tags_imprison = model(data[0], data[1])
                loss_accu = criterion_accu(outputs_accu, data[2])
                loss_article = criterion_article(outputs_article, data[3])
                loss_imprison = criterion_imprison(outputs_imprison, data[4])
                loss = loss_accu + loss_article + loss_imprison
                total_train_loss += loss.item()
                loss.backward()
                optimizer.step()
                eval_train_accu.evaluate(tags_accu, data[2].tolist())
                eval_train_article.evaluate(tags_article, data[3].tolist())
                eval_train_imprison.evaluate(tags_imprison, data[4].tolist())
            train_losses.append(total_train_loss)
            eval_train_accu.generate_result()
            eval_train_article.generate_result()
            eval_train_imprison.generate_result()
            print("train loss: %.3f" % total_train_loss)
            print("accu - micro, precision:%.3f, recall:%.3f, F1:%.3f, macro, precision:%.3f, recall:%.3f, F1:%.3f\narticle - micro, precision:%.3f, recall:%.3f, F1:%.3f, macro, precision:%.3f, recall:%.3f, F1:%.3f\nimprison - micro, precision:%.3f, recall:%.3f, F1:%.3f, macro, precision:%.3f, recall:%.3f, F1:%.3f\n" % \
                                       (eval_train_accu.precision_micro,
                                       eval_train_accu.recall_micro,
                                       eval_train_accu.F1_micro,
                                       eval_train_accu.precision_macro,
                                       eval_train_accu.recall_macro,
                                       eval_train_accu.F1_macro,

                                       eval_train_article.precision_micro,
                                       eval_train_article.recall_micro,
                                       eval_train_article.F1_micro,
                                       eval_train_article.precision_macro,
                                       eval_train_article.recall_macro,
                                       eval_train_article.F1_macro,

                                       eval_train_imprison.precision_micro,
                                       eval_train_imprison.recall_micro,
                                       eval_train_imprison.F1_micro,
                                       eval_train_imprison.precision_macro,
                                       eval_train_imprison.recall_macro,
                                       eval_train_imprison.F1_macro))

            # Valid
            model.eval()
            eval_valid_accu = Eval(len(self.list_accu))
            eval_valid_article = Eval(len(self.list_law))
            eval_valid_imprison = Eval(len(self.list_imprison))
            total_valid_loss = 0
            for data in valid_data:
                model.init_hidden()
                outputs_accu, outputs_article, outputs_imprison, tags_accu, tags_article, tags_imprison = model(data[0], data[1])
                loss_accu = criterion_accu(outputs_accu, data[2])
                loss_article = criterion_article(outputs_article, data[3])
                loss_imprison = criterion_imprison(outputs_imprison, data[4])
                loss = loss_accu + loss_article + loss_imprison
                total_valid_loss += loss.item()
                eval_valid_accu.evaluate(tags_accu, data[2].tolist())
                eval_valid_article.evaluate(tags_article, data[3].tolist())
                eval_valid_imprison.evaluate(tags_imprison, data[4].tolist())
            valid_losses.append(total_valid_loss)
            eval_valid_accu.generate_result()
            eval_valid_article.generate_result()
            eval_valid_imprison.generate_result()
            name = "bs%d_lr%f_es%d_hs%d_as%d_%s.png" % (
            batch_size, lr, embed_size, hidden_size, accu_hidden_size, self.train_set)
            epoch_list.set_description(name[0:-4])
            print("valid loss: %.3f" % total_valid_loss)
            print("accu - micro, precision:%.3f, recall:%.3f, F1:%.3f, macro, precision:%.3f, recall:%.3f, F1:%.3f\narticle - micro, precision:%.3f, recall:%.3f, F1:%.3f, macro, precision:%.3f, recall:%.3f, F1:%.3f\nimprison - micro, precision:%.3f, recall:%.3f, F1:%.3f, macro, precision:%.3f, recall:%.3f, F1:%.3f\n" % \
                                       (eval_valid_accu.precision_micro,
                                       eval_valid_accu.recall_micro,
                                       eval_valid_accu.F1_micro,
                                       eval_valid_accu.precision_macro,
                                       eval_valid_accu.recall_macro,
                                       eval_valid_accu.F1_macro,

                                       eval_valid_article.precision_micro,
                                       eval_valid_article.recall_micro,
                                       eval_valid_article.F1_micro,
                                       eval_valid_article.precision_macro,
                                       eval_valid_article.recall_macro,
                                       eval_valid_article.F1_macro,

                                       eval_valid_imprison.precision_micro,
                                       eval_valid_imprison.recall_micro,
                                       eval_valid_imprison.F1_micro,
                                       eval_valid_imprison.precision_macro,
                                       eval_valid_imprison.recall_macro,
                                       eval_valid_imprison.F1_macro))

            valid_accu_f1.append(eval_valid_accu.F1_macro)
            valid_article_f1.append(eval_valid_article.F1_macro)
            valid_imprison_f1.append(eval_valid_imprison.F1_macro)
            total_f1 = eval_valid_accu.F1_macro + eval_valid_article.F1_macro + eval_valid_imprison.F1_macro
            if total_f1 > best_f1:
                best_epoch = epoch
                best_f1 = total_f1
                self.config.set("train", "last_model", epoch)
            torch.save(model.state_dict(), os.path.join(model_path, "model-%d.pkl" % (epoch)))
            self.demo.train_valid_record(name, train_losses, valid_losses)

        return valid_accu_f1[best_epoch], valid_article_f1[best_epoch], valid_imprison_f1[best_epoch]
