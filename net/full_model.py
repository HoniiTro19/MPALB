import ipdb
import json
import torch
import torch.nn as nn
from path import Path
from net.rnn import RNN
import torch.nn.functional as F
from utils import load_list_dict
from net.forecast import Forecast
from configure import ConfigParser
from torch.autograd import Variable
from net.attention_fact import AttentionFact
from net.attention_article import AttentionArticle
from net.attention_legalbasis import AttentionLegalbasis

class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        config = ConfigParser()
        path = Path(config)
        use_gpu = config.getboolean("train", "use_gpu")
        device_num = config.getint("train", "gpu_device")
        cuda = torch.device("cuda:%d" % device_num)
        cpu = torch.device("cpu")
        device = cuda if use_gpu else cpu
        self.device = device

        cell_name = config.get("RNN", "cell_name")
        hidden_size = config.getint("train", "hidden_size")
        embed_size = config.getint("preprocess", "embed_size")
        criminal_basis_num = len(config.get("preprocess", "criminal_basis_ids").split(","))
        penalty_basis_num = len(config.get("preprocess", "penalty_basis_ids").split(","))
        fact_hidden_size = int(hidden_size / criminal_basis_num)
        article_hidden_size = int(hidden_size / (penalty_basis_num + 1))
        train_set = config.get("train", "train_set")
        is_remove_unfrequent = config.getboolean("preprocess", "is_remove_unfrequent")
        if is_remove_unfrequent:
            article_w2v_path = path.article_w2v_small_path if train_set == "small" else path.article_w2v_large_path
        else:
            article_w2v_path = path.article_w2v_path

        # Embedding
        embed_path = path.embed_matrix_path
        with open(embed_path, "r", encoding="UTF-8") as file:
            embed = json.loads(file.readline())
        embed = torch.tensor(embed, dtype=torch.float32)
        self.embed = nn.Embedding.from_pretrained(embed).requires_grad_(True)

        # RNN
        self.rnn_fact = RNN(cell_name, fact_hidden_size)
        self.rnn_article = RNN(cell_name, article_hidden_size)

        # Linear
        self.reduce_concate_hidden = nn.Linear(2 * hidden_size, hidden_size, bias=True)
        self.reduce_fact_hidden = nn.Linear(hidden_size, article_hidden_size, bias=True)

        # Attention
        self.criminal_basis_querys = torch.randn((1, fact_hidden_size), device=device, requires_grad=True)
        self.penalty_basis_querys = torch.randn((1, article_hidden_size), device=device, requires_grad=True)
        self.attention_fact = AttentionFact()
        self.attention_legalbasis = AttentionLegalbasis()
        self.attention_article = AttentionArticle()

        # Forecast
        self.forecast_article = Forecast("law", hidden_size)
        self.forecast_accu = Forecast("accu", hidden_size)
        self.forecast_imprison = Forecast("imprison", hidden_size)

        # Article, Criminal Basis, Penalty Basis
        criminal_basis_w2v_path = path.criminal_basis_w2v_path
        criminal_basis_w2v = list()
        criminal_basis_len = list()
        with open(criminal_basis_w2v_path, "r", encoding="UTF-8") as file:
            lines = json.loads(file.readline())
            for line in lines:
                criminal_basis_w2v.append(line["fact"])
                criminal_basis_len.append(line["len"])

        penalty_basis_w2v_path = path.penalty_basis_w2v_path
        penalty_basis_w2v = list()
        penalty_basis_len = list()
        with open(penalty_basis_w2v_path, "r", encoding="UTF-8") as file:
            lines = json.loads(file.readline())
            for line in lines:
                penalty_basis_w2v.append(line["fact"])
                penalty_basis_len.append(line["len"])

        article_w2v = list()
        article_len = list()
        with open(article_w2v_path, "r", encoding="UTF-8") as file:
            lines = json.loads(file.readline())
            for line in lines:
                article_w2v.append(line["fact"])
                article_len.append(line["len"])

        self.criminal_basis_w2v = torch.tensor(criminal_basis_w2v, device=device, dtype=torch.int64)
        self.criminal_basis_len = torch.tensor(criminal_basis_len, device=device, dtype=torch.int64)
        self.article_w2v = torch.tensor(article_w2v, device=device, dtype=torch.int64)
        self.article_len = torch.tensor(article_len, device=device, dtype=torch.int64)
        self.penalty_basis_w2v = torch.tensor(penalty_basis_w2v, device=device, dtype=torch.int64)
        self.penalty_basis_len = torch.tensor(penalty_basis_len, device=device, dtype=torch.int64)

        # Relu
        self.relu = nn.GELU()

    def forward(self, fact, seq_len):
        criminal_basis_embed = self.embed(self.criminal_basis_w2v)
        self.rnn_article.init_hidden(criminal_basis_embed.shape[0])
        criminal_basis_hidden = self.rnn_article(criminal_basis_embed, self.criminal_basis_len)
        criminal_basis_hidden = self.attention_legalbasis(criminal_basis_hidden, self.criminal_basis_querys)

        penalty_basis_embed = self.embed(self.penalty_basis_w2v)
        self.rnn_article.init_hidden(penalty_basis_embed.shape[0])
        penalty_basis_hidden = self.rnn_article(penalty_basis_embed, self.penalty_basis_len)
        penalty_basis_hidden = self.attention_legalbasis(penalty_basis_hidden, self.penalty_basis_querys)

        article_embed = self.embed(self.article_w2v)
        self.rnn_article.init_hidden(article_embed.shape[0])
        article_hidden = self.rnn_article(article_embed, self.article_len)

        fact_embed = self.embed(fact)
        self.rnn_fact.init_hidden()
        fact_rnn = self.rnn_fact(fact_embed, seq_len)
        fact_hidden = self.attention_fact(fact_rnn, criminal_basis_hidden)
        fact_reduced_hidden = self.reduce_fact_hidden(fact_hidden)

        outputs_accu, tags_accu = self.forecast_accu(fact_hidden)
        outputs_article, tags_article = self.forecast_article(fact_hidden)
        tensor_article = torch.tensor(tags_article, device=self.device, dtype=torch.int64)

        result_article = article_hidden[tensor_article]
        result_article = self.attention_article(result_article, (fact_reduced_hidden, penalty_basis_hidden))
        concate_hidden = self.relu(self.reduce_concate_hidden(torch.cat((result_article, fact_hidden), dim=-1)))
        outputs_imprison, tags_imprison = self.forecast_imprison(concate_hidden)
        return outputs_accu, outputs_article, outputs_imprison, tags_accu, tags_article, tags_imprison