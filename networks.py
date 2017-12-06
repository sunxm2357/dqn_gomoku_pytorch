import numpy as np
import torch
import torch.nn as nn
from math import floor
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

################## Functions #########################
def define_dqn(board_width, units, alpha=0.1, type='fc', layers=2, gpu_ids=[]):
    use_gpu = len(gpu_ids)

    if use_gpu:
        assert(torch.cuda.is_available())

    if type == 'fc':
        dqn = DQN_FC(board_width, units, gpu_ids=gpu_ids, alpha=alpha, layers=layers)
    elif type == 'conv':
        dqn = DQN_CONV(board_width, units, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError("the DQN's type [%s] is not implemented"%type)

    if len(gpu_ids) > 0:
        dqn.cuda(device_id=gpu_ids[0])
    return dqn


################## Classes ###########################
class DQN_FC(nn.Module):
    def __init__(self, width, units, gpu_ids, alpha=0.1, layers=2):
        super(DQN_FC, self).__init__()
        self.gpu_ids = gpu_ids

        # torch.nn.Linear(in_features, out_features, bias=True)
        in_feature = width**2
        self.linear1 = nn.Linear(in_feature, in_feature)
        self.lrelu1 = nn.LeakyReLU(negative_slope=alpha)

        layer = [self.linear1, self.lrelu1]
        for i in range(layers):
            if i == 0:
                layer.append(nn.Linear(in_feature, units))
            else:
                layer.append(nn.Linear(units, units))
            layer.append(nn.LeakyReLU(negative_slope=alpha))
            # torch.nn.Dropout(p=0.5, inplace=False)
            layer.append(nn.Dropout(p=0.2))

        layer.append(nn.Linear(units, in_feature))

        self.q_a = nn.Sequential(*layer)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):# TODO, turn all data to float
            q_action = nn.parallel.data_parallel(self.q_a, input, self.gpu_ids)
            return q_action
        else:
            q_action = self.q_a(input)
            return q_action


class DQN_CONV(nn.Module):
    def __init__(self, width, units, gpu_ids):
        super(DQN_CONV, self).__init__()
        self.gpu_ids = gpu_ids

        self.conv1 = nn.Conv2d(1, units, kernel_size=5, stride=2, padding=2)
        w = floor((width - 1) / 2. + 1)
        self.bn1 = nn.BatchNorm2d(units)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(units, 2*units, kernel_size=5, stride=2, padding=2)
        w = floor((w - 1) / 2. + 1)
        self.bn2 = nn.BatchNorm2d(2*units)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(2*units, 2*units, kernel_size=5, stride=2, padding=2)
        w = floor((w - 1) / 2. + 1)
        self.bn3 = nn.BatchNorm2d(2*units)
        self.relu3 = nn.ReLU()
        self.linear = nn.Linear(int(w)**2 * 2*units, width**2)
        layers = [self.conv1, self.bn1, self.relu1, self.conv2, self.bn2, self.relu2, self.conv3, self.bn3, self.relu3]
        # layers = [self.conv1, self.bn1, self.relu1]
        self.cnns = nn.Sequential(*layers)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):# TODO, turn all data to float
            x = nn.parallel.data_parallel(self.cnns, input, self.gpu_ids)
            x = x.view(x.size(0), -1)
            q_action = nn.parallel.data_parallel(self.linear, x, self.gpu_ids)
            return q_action
        else:
            # pdb.set_trace()
            x = self.cnns(input)
            # pdb.set_trace()
            x = x.view(x.size(0), -1)
            q_action = self.linear(x)
            return q_action
