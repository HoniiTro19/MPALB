import ipdb
import torch
import numpy as np
import torch.nn as nn
from configure import ConfigParser
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN(nn.Module):
    def __init__(self, cell_name, hidden_size):
        super(RNN, self).__init__()
        config = ConfigParser()
        use_gpu = config.getboolean("train", "use_gpu")
        device_num = config.getint("train", "gpu_device")
        cuda = torch.device("cuda:%d" % device_num)
        cpu = torch.device("cpu")
        self.device = cuda if use_gpu else cpu

        self.batch_size = config.getint("train", "batch_size")
        self.embed_size = config.getint("preprocess", "embed_size")
        self.hidden_size = hidden_size
        is_bidirection = config.getboolean("RNN", "is_bidirection")
        self.direction = 2 if is_bidirection else 1
        self.num_layers = config.getint("RNN", "num_layers")
        self.max_seq_len = config.getint("preprocess", "max_seq_len")
        self.cell_name = cell_name
        torch.manual_seed(config.getint("train", "random_seed"))
        if self.cell_name == "LSTM":
            self.rnn = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=is_bidirection)
        elif self.cell_name == "GRU":
            self.rnn = nn.GRU(self.embed_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=is_bidirection)
        else:
            raise Exception("Cannot recognize the cell name %s, supposed to be 'LSTM' or 'GRU'." % cell_name)

        # Linear
        self.reduce_hidden = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=True)

        # ReLU
        self.relu = nn.GELU()

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.cell_name == "LSTM":
            self.hidden = (
                torch.zeros((self.num_layers * self.direction, batch_size, self.hidden_size), device=self.device, requires_grad=True),
                torch.zeros((self.num_layers * self.direction, batch_size, self.hidden_size), device=self.device, requires_grad=True))
        else:
            self.hidden = torch.randn((self.num_layers * self.direction, batch_size, self.hidden_size), device=self.device, requires_grad=True)

    def forward(self, fact, seq_lens):
        """
        :param fact: [self.batch_size, doc_len, sent_len, embed_size]
        :param seq_len: [self.batch_size, doc_len]
        :return: [self.batch_size, doc_len, self.hidden_size]
        """
        seq_lens = seq_lens.tolist()
        fact_packed = pack_padded_sequence(fact, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_out, self.hidden = self.rnn(fact_packed, self.hidden)
        fact_unpacked, _ = pad_packed_sequence(rnn_out, batch_first=True, total_length=self.max_seq_len)
        rnn_reduced = self.relu(self.reduce_hidden(fact_unpacked))
        return rnn_reduced
