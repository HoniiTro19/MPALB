import os
import ipdb
import torch
from path import Path
from eval import Eval
from demo import Demo
import torch.nn as nn
from loader import SmallDataset
from utils import load_list_dict
from configure import ConfigParser
from net.full_model import FullModel
from torch.utils.data import DataLoader

config = ConfigParser()
path = Path(config)
demo = Demo()

model_path = path.model_path
model = FullModel()
model_trained_path = model_path + "model-%d.pkl" % config.getint("train", "last_model")
if os.path.exists(model_trained_path):
    model.load_state_dict(torch.load(model_trained_path))
else:
    raise Exception("No trained model accessible.")

use_gpu = config.getboolean("train", "use_gpu")
device_num = config.getint("train", "gpu_device")
cuda = torch.device("cuda:%d" % device_num)
cpu =  torch.device("cpu")
if use_gpu:
    if not torch.cuda.is_available():
        raise Exception("Cuda device is not available")
    device = cuda
else:
    device = cpu
model.to(device)
model.eval()

train_set = config.get("train", "train_set")
batch_size = config.getint("train", "batch_size")
is_shuffle = config.getboolean("train", "is_shuffle")
if train_set == "small":
    test_small_w2v_path = path.test_small_w2v_path
    test_dataset = SmallDataset(test_small_w2v_path)
    test_data = DataLoader(test_dataset, batch_size, is_shuffle, drop_last=True)
elif train_set == "large":
    raise Exception("Not ready for large dataset")
else:
    raise Exception("Cannot recognize the dataset type %s, suppose to be 'small' or 'large'" % train_set)

criterion = nn.CrossEntropyLoss().to(device)
is_remove_unfrequent = config.getboolean("preprocess", "is_remove_unfrequent")
train_set = config.get("train", "train_set")
if is_remove_unfrequent:
    accu_path = path.meta_accu_frequent_small_path if train_set == "small" else path.meta_accu_frequent_large_path
    law_path = path.meta_law_frequent_small_path if train_set == "small" else path.meta_law_frequent_large_path
else:
    accu_path = path.meta_accu_path
    law_path = path.meta_law_path
imprison_path = path.meta_imprison_path

list_accu, dict_accu = load_list_dict(accu_path)
list_law, dict_article = load_list_dict(law_path)
list_imprison, dict_imprison = load_list_dict(imprison_path)

eval_accu = Eval(len(list_accu))
eval_article = Eval(len(list_law))
eval_imprison = Eval(len(list_imprison))
total_loss = 0
for data in test_data:
    outputs_accu, outputs_article, outputs_imprison, tags_accu, tags_article, tags_imprison = model(data[0], data[1])
    loss_accu = criterion(outputs_accu, data[2])
    loss_article = criterion(outputs_article, data[3])
    loss_imprison = criterion(outputs_imprison, data[4])
    loss = loss_accu + loss_article + loss_imprison
    total_loss += loss.item()
    eval_accu.evaluate(tags_accu, data[2].tolist())
    eval_article.evaluate(tags_article, data[3].tolist())
    eval_imprison.evaluate(tags_imprison, data[4].tolist())
eval_accu.generate_result()
eval_article.generate_result()
eval_imprison.generate_result()
print("test loss: %.4f" % total_loss)
print("accu - micro, precision:%.4f, recall:%.4f, F1:%.4f, macro, precision:%.4f, recall:%.4f, F1:%.4f\narticle - micro, precision:%.4f, recall:%.4f, F1:%.4f, macro, precision:%.4f, recall:%.4f, F1:%.4f\nimprison - micro, precision:%.4f, recall:%.4f, F1:%.4f, macro, precision:%.4f, recall:%.4f, F1:%.4f\n" % \
                           (eval_accu.precision_micro,
                           eval_accu.recall_micro,
                           eval_accu.F1_micro,
                           eval_accu.precision_macro,
                           eval_accu.recall_macro,
                           eval_accu.F1_macro,

                           eval_article.precision_micro,
                           eval_article.recall_micro,
                           eval_article.F1_micro,
                           eval_article.precision_macro,
                           eval_article.recall_macro,
                           eval_article.F1_macro,

                           eval_imprison.precision_micro,
                           eval_imprison.recall_micro,
                           eval_imprison.F1_micro,
                           eval_imprison.precision_macro,
                           eval_imprison.recall_macro,
                           eval_imprison.F1_macro))

demo.display_imprison(eval_imprison.confusion_matrix, "imprison_confusion_matrix.png")

