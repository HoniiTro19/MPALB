import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def chinese2number(chinese):
    chinese = chinese.strip()
    char2number = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 1e1, "百": 1e2, "千": 1e3,
                   "万": 1e4, "亿": 1e8}
    number = 0
    temp = 0
    for i in chinese:
        if i not in char2number.keys():
            return chinese

    for char in chinese:
        if char in ["十", "百", "千", "万", "亿"]:
            if temp == 0:
                temp = 1
            temp *= char2number[char]
            number += temp
            temp = 0
        elif char == "零":
            continue
        else:
            temp = char2number[char]

    if temp != 0:
        number += temp

    return int(number)

def load_list_dict(list_path):
    with open(list_path, "r", encoding="UTF-8") as file:
        item_list = [item.strip("\n") for item in file.readlines()]
    idx = 0
    item_dict = dict()
    for item in item_list:
        item_dict[item] = idx
        idx += 1
    return item_list, item_dict


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss