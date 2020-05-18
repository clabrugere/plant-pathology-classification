import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0)
    if isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)

class BasicWideBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout_p=.5):
        super(BasicWideBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=1)
        
        if in_channels != out_channels or stride != 1:
            self.shunt = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride)
        else:
            self.shunt = nn.Identity()
    
    def forward(self, x):
        # x: (batch_size, in_channels, H, W)
        
        # first convolution
        out = self.conv1(self.activation(self.bn1(x)))
        
        # dropout
        out = self.dropout(out)
        
        # second convolution
        out = self.conv2(self.activation(self.bn2(out)))
        
        # return residual
        out += self.shunt(x)
        
        # out: (batch_size, out_channels, (H - kernel_size - 1) / stride + 1, (W - kernel_size - 1) / stride + 1)
        return out


class WideResNet(nn.Module):
    
    def __init__(self, in_channels, first_out_channels, n_classes, k, n, dropout_p=.5):
        super(WideResNet, self).__init__()
        
        # first layer is a convolution
        self.conv1 = nn.Conv2d(in_channels, first_out_channels, kernel_size=3, padding=1)
        self._in_channels = first_out_channels
        
        # build network stages
        self.stage1 = self._make_stage(n, first_out_channels*k, kernel_size=3, stride=1, dropout_p=dropout_p)
        self.stage2 = self._make_stage(n, first_out_channels*2*k, kernel_size=3, stride=2, dropout_p=dropout_p)
        self.stage3 = self._make_stage(n, first_out_channels*4*k, kernel_size=3, stride=2, dropout_p=dropout_p)
        
        self.bn = nn.BatchNorm2d(first_out_channels*4*k)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(first_out_channels*4*k, n_classes)
    
    def _make_stage(self, n, out_channels, kernel_size, stride, dropout_p):
        # reduce (H,W) with the stride on the first layer only
        strides = np.ones(n, dtype=np.int16)
        strides[0] = stride
        blocks = []
        
        for i, stride in enumerate(strides):
            blocks.append(BasicWideBlock(self._in_channels, out_channels, kernel_size=kernel_size, stride=stride, dropout_p=dropout_p))
            
            # update the number of channels once for the rest of the blocks
            if i == 0:
                self._in_channels = out_channels
            
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        
        # first convolution
        out = self.conv1(x)
        
        # pass through stages
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        
        # bn and activation
        out = self.activation(self.bn(out))
        
        # pool, reshape and pass through dense
        out = self.pool(out)
        out = torch.squeeze(out)
        out = self.linear(out)
        
        return out
    
