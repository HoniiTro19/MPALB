import ipdb
import torch
import torch.nn as nn
from configure import ConfigParser
import torch.nn.functional as F

class AttentionFact(nn.Module):
    def __init__(self):
        config = ConfigParser()
        self.is_single_process = config.getboolean("demo", "is_single_process")
        super(AttentionFact, self).__init__()

    def forward(self, hidden, querys):
        '''
        :param hidden: [batch_size, seq_len, hidden_size]
        :param querys: [channel_num, hidden_size]
        :return: [batch_size, hidden_size]
        '''
        assert hidden.shape[-1] == querys.shape[-1]
        hidden_size = querys.shape[-1]
        channel_num = querys.shape[0]
        querys = querys.transpose(0, 1).contiguous()
        factor = torch.matmul(hidden, querys)
        factor = factor.transpose(1, 2).contiguous()
        factor = F.softmax(factor, dim=2)
        if self.is_single_process:
            print("fact attention->", factor)
        result = torch.matmul(factor, hidden)
        result = result.view(-1, channel_num * hidden_size)
        return result