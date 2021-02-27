import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionArticle(nn.Module):
    def __init__(self):
        super(AttentionArticle, self).__init__()

    def forward(self, hidden, querys):
        '''
        article
        :param hidden: [batch_size, seq_len, hidden_size]
        :param querys: ([batch_size, hidden_size]，[channel_num, hidden_size])
        :return: [batch_size, hidden_size]
        '''
        fact_querys = querys[0]
        penalty_basis_query = querys[1]
        assert fact_querys.shape[-1] == penalty_basis_query.shape[-1] and fact_querys.shape[-1] == hidden.shape[-1]
        batch_size = fact_querys.shape[0]
        penalty_basis_querys = [penalty_basis_query] * batch_size
        penalty_basis_querys = torch.stack(penalty_basis_querys)
        fact_querys = torch.unsqueeze(fact_querys, dim=1)
        querys = torch.cat((fact_querys, penalty_basis_querys), dim=1)
        hidden = hidden.transpose(1, 2).contiguous()
        factor = torch.bmm(querys, hidden)
        factor = F.softmax(factor, dim=2)
        hidden = hidden.transpose(1, 2).contiguous()
        result = torch.bmm(factor, hidden)
        result = result.view(batch_size, -1)
        return result