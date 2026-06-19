import torch
import numpy as np
import torch.nn.functional as F
import argparse
from torch.autograd import Function
parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])
args.device = 'cuda'
if torch.cuda.is_available():
    args.device = 'cuda:0'


class ResidualBlock(torch.nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.ReflectionPad1d(1),
            torch.nn.Conv1d(features, features, 3),
            torch.nn.InstanceNorm1d(features),
            torch.nn.ReLU(inplace=True),
            torch.nn.ReflectionPad1d(1),
            torch.nn.Conv1d(features, features, 3),
            torch.nn.InstanceNorm1d(features)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(torch.nn.Module):
    def __init__(self, input_nc=1, output_nc=1, n_residual_blocks=4):
        super(Generator, self).__init__()

        model = [
            torch.nn.ReflectionPad1d(3),
            torch.nn.Conv1d(input_nc, 64, 7),
            torch.nn.InstanceNorm1d(64),
            torch.nn.ReLU(inplace=True)
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                torch.nn.Conv1d(in_features, out_features, 3, stride=2, padding=1),
                torch.nn.InstanceNorm1d(out_features),
                torch.nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                torch.nn.ConvTranspose1d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                torch.nn.InstanceNorm1d(out_features),
                torch.nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [
            torch.nn.ReflectionPad1d(3),
            torch.nn.Conv1d(64, output_nc, 7),
            torch.nn.Tanh()
        ]

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(torch.nn.Module):
    def __init__(self, input_nc=1):
        super(Discriminator, self).__init__()
        model = [
            torch.nn.Conv1d(input_nc, 64, 4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv1d(64, 128, 4, stride=2, padding=1),
            torch.nn.InstanceNorm1d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv1d(128, 256, 4, stride=2, padding=1),
            torch.nn.InstanceNorm1d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv1d(256, 512, 4, stride=2, padding=1),
            torch.nn.InstanceNorm1d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv1d(512, 1, 4, padding=1)
        ]

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return torch.nn.functional.avg_pool1d(x, x.size()[2:]).view(x.size()[0], -1)

class CycleGAN(torch.nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(CycleGAN, self).__init__()

        self.G_XY = Generator(input_nc, output_nc)
        self.G_YX = Generator(output_nc, input_nc)

        self.D_X = Discriminator(input_nc)
        self.D_Y = Discriminator(output_nc)

    def forward(self, twin_data, real_data):
        # twin_data -- x
        # real_data -- y
        x2y = self.G_XY(twin_data)
        x2y2x = self.G_YX(x2y)

        y2x = self.G_YX(real_data)
        y2x2y = self.G_XY(y2x)

        r_x = self.G_YX(twin_data)
        r_y = self.G_XY(real_data)

        dx = self.D_X(twin_data)
        dy2x = self.D_X(y2x)

        dy = self.D_Y(real_data)
        dx2y = self.D_Y(x2y)

        return x2y, x2y2x, y2x2y, r_x, r_y, dx, dy2x, dy, dx2y


def main():
    model = CycleGAN().to(args.device)
 
if __name__ == '__main__':
     main()

if __name__ == '__main__':
     main()
