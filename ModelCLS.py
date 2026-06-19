import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import argparse
from torch.autograd import Function
parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])
args.device = 'cuda'
if torch.cuda.is_available():
    args.device = 'cuda:0'


class ECAAttention1D(nn.Module):

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.kernel_size = self._get_kernel_size(channels, gamma, b)
        self.avg_pool = nn.AdaptiveAvgPool1d(1) 
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size,
                              padding=self.kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def _get_kernel_size(self, channels, gamma, b):
        k = int(abs((math.log2(channels) + b) / gamma))
        return k if k % 2 else k + 1

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.transpose(1, 2)  # [batch, 1, channels
        y = self.conv(y)
        y = y.transpose(1, 2)
        y = self.sigmoid(y)

        return x * y


class CNN(torch.nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=128)
        self.max_pool1 = torch.nn.MaxPool1d(kernel_size=4)
        self.conv2 = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=64)
        self.max_pool2 = torch.nn.MaxPool1d(kernel_size=4)
        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16)
        self.max_pool3 = torch.nn.MaxPool1d(kernel_size=4)

        self.conv4 = torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=128)
        self.max_pool4 = torch.nn.MaxPool1d(kernel_size=4)
        self.conv5 = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=64)
        self.max_pool5 = torch.nn.MaxPool1d(kernel_size=4)
        self.conv6 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16)
        self.max_pool6 = torch.nn.MaxPool1d(kernel_size=4)

        self.attention = ECAAttention1D(channels=64)

        self.flatten = torch.nn.Flatten()


        self.lin1 = torch.nn.Linear(1408, 64)
        self.lin2 = torch.nn.Linear(64, 4)

    def forward(self, data, spectrum):
        x = F.relu(self.conv1(data))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.relu(self.conv3(x))
        x = self.max_pool3(x)
        x = self.attention(x)
        x = self.flatten(x)

        f = F.relu(self.conv4(spectrum))
        f = self.max_pool4(f)
        f = F.relu(self.conv5(f))
        f = self.max_pool5(f)
        f = F.relu(self.conv6(f))
        f = self.max_pool6(f)
        f = self.attention(f)
        f = self.flatten(f)
        feature = torch.concat([x, f], dim=1)

        output = F.relu(self.lin1(x))
        output = F.dropout(output, p=0.2, training=self.training)
        output = F.softmax(self.lin2(output), dim=-1)

        return output


def main():
    model = CNN(args).to(args.device)



if __name__ == '__main__':
     main()
