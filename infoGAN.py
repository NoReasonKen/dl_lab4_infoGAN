#!/usr/local/anaconda3/bin/python3

import numpy as np
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
BATCH_SIZE=64
NOISE_CON=54
NOISE_DIS=10
NOISE_SIZE=NOISE_CON + NOISE_DIS
IMAGE_SIZE=NOISE_SIZE
LR_D=2e-4
LR_Q=1e-3
EPOCH=1

MNIST_PATH="./data"
MNIST_DOWNLOAD=False

MODE='train'
GPU_NUM=1
#===================================================================
class Gpart(nn.Module):
    def __init__(self, gpu):
        super().__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(NOISE_SIZE, 512, 4, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False),
                nn.Tanh())

        self.gpu = gpu

    def forward(self, x):
        if self.gpu > 1:
            out = nn.parallel.data_parallel(self.main, x, range(self.gpu))
        else:
            out = self.main(x)
        return out

class Dpart(nn.Module):
    def __init__(self, gpu):
        super().__init__()
        self.main = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, padding=1, bias=False),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(512, 1, 4, 1, bias=False),
                nn.Sigmoid())

        self.gpu = gpu

    def forward(self, x):
        if self.gpu > 1:
            out = nn.parallel.data_parallel(self.main, x, range(self.gpu))
        else:
            out = self.main(x)
        return out.view(-1)

class Qpart(nn.Module):
    def __init__(self, gpu):
        super().__init__()
        self.main = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, padding=1, bias=False),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, True))
        self.q_part = nn.Sequential(
                nn.Linear(8192, 100),
                nn.ReLU(True),
                nn.Linear(100, 10))

        self.gpu = gpu

    def forward(self, x):
        bs = x.size(0)
        if self.gpu > 1:
            out = nn.parallel.data_parallel(self.main, x, range(self.gpu))
            out = out.view(bs, -1, 1, 1)
            out = nn.parallel.data_parallel(self.q_part, out, range(self.gpu))
        else:
            out = self.main(x).view(bs, -1)
            out = self.q_part(out)

        return out
#===================================================================
def weights_init(model):
    name = model.__class__.__name__
    if name.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif name.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)

def train(epoch):
    print("Epoch: ", epoch)

    loss_D = 0
    loss_G = 0
    for idx, (data, _) in enumerate(train_loader):
        bs = data.size(0)
        label_noise = np.random.randint(NOISE_DIS, size=bs)
        onehot = np.zeros((bs, NOISE_DIS))
        onehot[range(bs), label_noise] = 1
        label_noise = torch.from_numpy(label_noise).type(torch.LongTensor).cuda()
        onehot = torch.from_numpy(onehot).type(torch.FloatTensor).cuda()
        noise = torch.Tensor(bs, NOISE_CON).cuda().uniform_(-1, 1)
        noise = torch.cat((noise, onehot), 1).view(-1, NOISE_SIZE, 1, 1)
        data = data.cuda()


        optimD.zero_grad()

        out_D_real = modelD(V(data))
        label = torch.ones(bs).cuda()
        loss_D_real = criterionD(out_D_real, V(label))
        loss_D += loss_D_real.data
        loss_D_real.backward()

        out_G = modelG(V(noise))
        out_G_copy = V(torch.Tensor(out_G.cpu().data).cuda())
        out_D_noise = modelD(out_G_copy)
        label = torch.zeros(bs).cuda()
        loss_D_noise = criterionD(out_D_noise, V(label))
        loss_D += loss_D_noise.data
        loss_D_noise.backward() 

        optimD.step()

               
        optimG.zero_grad()
        
        out_D = modelD(out_G)
        label = torch.ones(bs).cuda()
        loss_D_noise = criterionD(out_D, V(label))
        loss_G += loss_D_noise.data

        out_Q = modelQ(out_G)
        loss_Q = critetionQ(out_Q, V(label_noise))
        loss_G += loss_Q.data

        loss_G_noise = loss_D_noise + loss_Q
        loss_G_noise.backward()

        optimG.step()

        progress_bar(idx, len(train_loader),
                    'DLoss: %.3f, GLoss: %.3f' 
                    % ((loss_D/(idx+1)), (loss_G/(idx+1))))

def load_model():
    modelG = torch.load('model.pkl').cuda()

    label_noise = np.random.randint(NOISE_DIS, size=BATCH_SIZE)
    onehot = np.zeros((BATCH_SIZE, NOISE_DIS))
    onehot[range(BATCH_SIZE), label_noise] = 1
    onehot = torch.from_numpy(onehot).type(torch.FloatTensor).cuda()
    noise = torch.Tensor(BATCH_SIZE, NOISE_CON).cuda().uniform_(-1, 1)
    noise = V(torch.cat((noise, onehot), 1)).view(-1, NOISE_SIZE, 1, 1)

    sample = modelG(noise).cpu()
    save_image(sample.data, 'output.jpg', nrow=10)

if __name__ == '__main__':
    T = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                            transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(MNIST_PATH, train=True, download=MNIST_DOWNLOAD
                        , transform=T),
            batch_size=BATCH_SIZE,
            shuffle=True, num_workers=4, 
            )
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(MNIST_PATH, train=False, download=MNIST_DOWNLOAD
                        , transform=T),
            batch_size=100,
            shuffle=True, num_workers=4, 
            )

    cudnn.benchmark = True

    modelG = Gpart(GPU_NUM).cuda().apply(weights_init)
    modelD = Dpart(GPU_NUM).cuda().apply(weights_init)
    modelQ = Qpart(GPU_NUM).cuda().apply(weights_init)

    criterionD = nn.BCELoss().cuda()
    critetionQ = nn.CrossEntropyLoss().cuda()

    optimD = optim.Adam(modelD.parameters(), lr=LR_D, betas=(0.5, 0.99))
    optimG = optim.Adam([{'params':modelG.parameters()}, {'params':modelQ.parameters()}]
                        , lr=LR_Q, betas=(0.5, 0.99))

    if MODE == 'load':
        load_model()
    else:
        for epoch in range(EPOCH):
            train(epoch)
        torch.save(modelG, 'model.pkl')
