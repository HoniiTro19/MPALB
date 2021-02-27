import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLegalbasis(nn.Module):
    def __init__(self):
        super(AttentionLegalbasis, self).__init__()

    def forward(self, hidden, querys):
        '''
        :param hidden: [batch_size, seq_len, hidden_size]
        :param querys: [1, hidden_size]
        :return: [batch_size, hidden_size]
        '''
        assert hidden.shape[-1] == querys.shape[-1]
        querys = querys.transpose(0, 1).contiguous()
        factor = torch.matmul(hidden, querys)
        factor = factor.transpose(1, 2).contiguous()
        factor = F.softmax(factor, dim=2)
        result = torch.matmul(factor, hidden)
        result = torch.squeeze(result)
        return result