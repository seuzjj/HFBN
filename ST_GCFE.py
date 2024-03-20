import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=2):#num_subset=3
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        if True:
            self.edge_importance = nn.Parameter(torch.ones(self.A.size()))
        else:
            self.edge_importance = 1

        self.num_subset = num_subset
        self.conv_sp = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_sp.append(nn.Conv2d(in_channels, out_channels, 1))
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_sp[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A*self.edge_importance
        A = self.PA+A

        y = None
        for i in range(self.num_subset):
            A=A.view(1,V,V)
            A2 = x.view(N, C * T, V)
            z = self.conv_sp[i](torch.matmul(A2, A).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True,kernel=6,num_set=2):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, num_subset=num_set)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride,kernel_size=kernel)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=kernel, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)
def edge2mat(num_node):
    self_link = [(i, i) for i in range(num_node)]
    A = np.zeros((num_node, num_node))
    for i, j in self_link:
        A[j, i] = 1
    return A

class Model(nn.Module):
    def __init__(self, num_class=2, num_point=90, num_person=6,  in_channels=1,kernel=1,num_set=2):
        super(Model, self).__init__()

        A = edge2mat(num_point)

        # A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(1, 12, A, residual=False,kernel=kernel,num_set=num_set)
        self.l2 = TCN_GCN_unit(12, 12, A,kernel=kernel,num_set=num_set)
        self.l4 = TCN_GCN_unit(12, 12, A,kernel=kernel,num_set=num_set)
        self.l5 = TCN_GCN_unit(12, 32, A, stride=2,kernel=kernel,num_set=num_set)
        self.l6 = TCN_GCN_unit(32, 32, A,kernel=kernel,num_set=num_set)
        self.l8 = TCN_GCN_unit(32, 64, A, stride=2,kernel=kernel,num_set=num_set)


        self.fc = nn.Linear(64, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l8(x)
        c_new = x.size(1)
        x = x.mean(2)
        return x
