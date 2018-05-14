#!/usr/bin/python3

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
EPOCH=80

MNIST_PATH="./data"
MNIST_DOWNLOAD=False

MODE='train'
MULTIGPU=True
#===================================================================
class Gpart(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        out = self.main(x)
        return out

class Dpart(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        out = self.main(x)
        return out.view(-1)

class Qpart(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        bs = x.size(0)
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
    loss_D_fake = 0
    loss_Q_fake = 0
    for idx, (data, _) in enumerate(train_loader):
        bs = data.size(0)
        label_noise = np.random.randint(NOISE_DIS, size=bs)
        onehot = np.zeros((bs, NOISE_DIS))
        onehot[range(bs), label_noise] = 1
        label_noise = torch.from_numpy(label_noise).type(torch.LongTensor).cuda()
        onehot = torch.from_numpy(onehot).type(torch.FloatTensor).cuda()
        noise = torch.Tensor(bs, NOISE_CON).cuda().normal_()
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
        loss_D_fake += loss_D_noise.data

        out_Q = modelQ(out_G)
        loss_Q = critetionQ(out_Q, V(label_noise))
        loss_Q_fake += loss_Q.data

        loss_G_noise = loss_D_noise + loss_Q
        loss_G_noise.backward()

        optimG.step()

        progress_bar(idx, len(train_loader),
                    'DLoss: %.3f, DFLoss: %.3f, QLoss: %.3f, GLoss: %.3f' 
                    % ((loss_D/(idx+1)), (loss_D_fake/(idx+1)), 
                        (loss_Q_fake/(idx+1)), (loss_G_noise/(idx+1))))

def load_model(maxi):
    for i in range(1, maxi):
        modelG = torch.load('model'+i+'0.pkl').cuda()
        
        label_noise = np.arange(NOISE_DIS)
        label_noise = np.tile(label_noise, 10)
        onehot = np.zeros((10 * NOISE_DIS, NOISE_DIS))
        onehot[range(10 * NOISE_DIS), label_noise] = 1
        onehot = torch.from_numpy(onehot).type(torch.FloatTensor)
        noise = torch.Tensor(10, NOISE_CON).normal_().numpy()
        noise = torch.from_numpy(np.repeat(noise, 10, axis=0))
        noise = torch.cat((noise, onehot), 1).view(-1, NOISE_SIZE, 1, 1).cuda()

        sample = modelG(noise).cpu()
        save_image(sample.data, 'output'+i+'0.jpg', nrow=10)

if __name__ == '__main__':
    if MODE == 'load':
        load_model(EPOCH / 10)
    else:
        T = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(MNIST_PATH, train=True, download=MNIST_DOWNLOAD
                            , transform=T),
                batch_size=BATCH_SIZE,
                shuffle=True, num_workers=4, 
                )

        cudnn.benchmark = True

        modelG = Gpart().cuda()
        modelD = Dpart().cuda()
        modelQ = Qpart().cuda()

        if MULTIGPU:
            modelG = torch.nn.DataParallel(modelG.cuda(), device_ids=[0])
            modelD = torch.nn.DataParallel(modelD.cuda(), device_ids=[0])
            modelQ = torch.nn.DataParallel(modelQ.cuda(), device_ids=[0])

        criterionD = nn.BCELoss().cuda()
        critetionQ = nn.CrossEntropyLoss().cuda()

        optimD = optim.Adam(modelD.parameters(), lr=LR_D, betas=(0.5, 0.99))
        optimG = optim.Adam([{'params':modelG.parameters()}, {'params':modelQ.parameters()}]
                            , lr=LR_Q, betas=(0.5, 0.99))

        for epoch in range(EPOCH):
            train(epoch)
            if epoch % 10 == 0 and epoch != 0:
                torch.save(modelG, 'model' + str(epoch) + '.pkl')
        
        torch.save(modelG, 'model80.pkl')
