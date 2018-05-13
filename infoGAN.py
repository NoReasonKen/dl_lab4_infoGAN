#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable as V
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils import progress_bar
#===================================================================
NOISE_CON=54
NOISE_DIS=10
NOISE_SIZE=NOISE_CON + NOISE_DIS
IMAGE_SIZE=NOISE_SIZE

MNIST_PATH="./data"
MNIST_DOWNLOAD=False
#===================================================================
class Gpart(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(NOISE_SIZE, 512, 4, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace),
                nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace),
                nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace),
                nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace),
                nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False),
                nn.Tanh())

    def forward(self, x):
        out = self.main(x)
        return out

class DQpart(nn.Module):
    def __init__(self, needQ):
        super().__init__()
        self.shared = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace),
                nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
                nn.batchNorm2d(128),
                LeakyReLU(0.2, inplace),
                nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
                nn.batchNorm2d(256),
                LeakyReLU(0.2, inplace),
                nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
                nn.batchNorm2d(512),
                LeakyReLU(0.2, inplace))
        self.d_part = nn.Sequential(
                nn.Conv2d(512, 1, 4, 1, bias=False),
                nn.Sigmoid())
        self.q_part = nn.Sequential(
                nn.Linear(8192, 100),
                nn.ReLU(),
                nn.Linear(100, 10))

    def forward(x):
        out = self.shared(x)
        out_d = self.d_part(out)
        out_q = self.q_part(out)
        return out_d, out_q

def weights_init(model):
    name = model.__class__.__name__
    if name.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif name.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
#===================================================================
if __name__ == '__main__':
    T = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                            transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(MNIST_PATH, train=True, download=MNIST_DOWNLOAD
                        , transform=T),
            batch_size=TRAIN_DATA_BATCH_SIZE,
            shuffle=True, num_workers=4, 
            )
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(MNIST_PATH, train=False, download=MNIST_DOWNLOAD
                        , transform=T),
            batch_size=100,
            shuffle=True, num_workers=4, 
            )

    cudnn.benchmark = True
