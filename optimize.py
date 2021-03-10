from demo import Demo
from train import Train
from preprocessor import Preprocessor

demo = Demo()
train = Train()
preprocessor = Preprocessor()
default_parameters = {"batch_size": 32, "embed_size": 100, "lr": 0.001, "hidden_size": 512}
batch_sizes = [16, 32, 64, 128]
embed_sizes = [50, 75, 100, 125, 150]
lrs = [0.0006, 0.0008, 0.001, 0.0012, 0.0014]
hidden_sizes = [256, 512, 1024, 2048]

# Batch Size
batchsize_accu_f1, batchsize_article_f1, batchsize_imprison_f1 = list(), list(), list()
for batch_size in batch_sizes:
    train.config.set("train", "batch_size", batch_size)
    best_accu_f1, best_article_f1, best_imprison_f1 = train.train_file()
    batchsize_accu_f1.append(best_accu_f1)
    batchsize_article_f1.append(best_article_f1)
    batchsize_imprison_f1.append(best_imprison_f1)
demo.optimize_plot("batch_size_optimize.png", batchsize_accu_f1, batchsize_article_f1, batchsize_imprison_f1, batch_sizes)
train.config.set("train", "batch_size", default_parameters["batch_size"])
#
# # Learning Rate
# lr_accu_f1, lr_article_f1, lr_imprison_f1 = list(), list(), list()
# for lr in lrs:
#     train.config.set("train", "lr", lr)
#     best_accu_f1, best_article_f1, best_imprison_f1 = train.train_file()
#     lr_accu_f1.append(best_accu_f1)
#     lr_article_f1.append(best_article_f1)
#     lr_imprison_f1.append(best_imprison_f1)
# demo.optimize_plot("learning_rate_optimize.png", lr_accu_f1, lr_article_f1, lr_imprison_f1, lrs)
# train.config.set("train", "lr", default_parameters["lr"])

# Hidden Size
hiddensize_accu_f1, hiddensize_article_f1, hiddensize_imprison_f1 = list(), list(), list()
for hidden_size in hidden_sizes:
    train.config.set("train", "hidden_size", hidden_size)
    best_accu_f1, best_article_f1, best_imprison_f1 = train.train_file()
    hiddensize_accu_f1.append(best_accu_f1)
    hiddensize_article_f1.append(best_article_f1)
    hiddensize_imprison_f1.append(best_imprison_f1)
demo.optimize_plot("hidden_size_optimize.png", hiddensize_accu_f1, hiddensize_article_f1, hiddensize_imprison_f1, hidden_sizes)
train.config.set("train", "hidden_size", default_parameters["hidden_size"])

# # Embed Size
# embedsize_accu_f1, embedsize_article_f1, embedsize_imprison_f1 = list(), list(), list()
# for embed_size in embed_sizes:
#     train.config.set("preprocess", "embed_size", embed_size)
#     preprocessor = Preprocessor()
#     preprocessor.preprocess()
#     best_accu_f1, best_article_f1, best_imprison_f1 = train.train_file()
#     embedsize_accu_f1.append(best_accu_f1)
#     embedsize_article_f1.append(best_article_f1)
#     embedsize_imprison_f1.append(best_imprison_f1)
# demo.optimize_plot("embed_size_optimize.png", embedsize_accu_f1, embedsize_article_f1, embedsize_imprison_f1, embed_sizes)
# train.config.set("preprocess", "embed_size", default_parameters["embed_size"])
