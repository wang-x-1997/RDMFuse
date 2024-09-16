#coding=utf-8
from __future__ import print_function
import argparse
import os
from dataset import fusiondata
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim as optim
from net1 import DecNet
import pytorch_msssim
from TD import TD
import kornia
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Training settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', type=str, default='data',help='facades')
parser.add_argument('--batchSize', type=int, default=3, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.001 , help='Learning Rate. Default=0.0002')#0.000005
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=150, help='weight on L1 term in objective')
parser.add_argument('--alpha', type=int, default=0.25)
opt = parser.parse_args()


use_cuda= not opt.cuda and torch.cuda.is_available()

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)
    

device=torch.device("cuda" if use_cuda else "cpu")

print('===> Loading datasets')
root_path = "data/"

dataset = fusiondata(os.path.join(root_path,opt.dataset))

training_data_loader = DataLoader(dataset=dataset, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=None)

print('===> Building model')

model_Dec = DecNet()
model_Dec = model_Dec.cuda()

def downsample(image):
    return F.avg_pool2d(image, kernel_size=32, stride=16, padding=0)


def gradient(input_tensor, direction):
    smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
    smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)

    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                  stride=1, padding=1))
    return grad_out

def mutual_i_loss(inputI):

    return torch.mean(gradient(inputI, "x") * torch.exp(-10 * gradient(inputI, "x")) +
                          gradient(inputI, "y") * torch.exp(-10 * gradient(inputI, "y")))

def mutual_loss(ill,VI):
    ill_x  = gradient(ill,'x')
    ill_y = gradient(ill, 'y')
    VI_x  = gradient(VI,'x')
    VI_y = gradient(VI, 'y')
    x_loss = abs(ill_x/ VI_x.clamp_min(1e-10) )
    y_loss = abs(ill_y / VI_y.clamp_min(1e-10))
    return torch.mean(x_loss+y_loss)

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self,x,weight_map=None):
        self.h_x = x.size()[2]
        self.w_x = x.size()[3]
        self.batch_size = x.size()[0]
        if weight_map is None:
            self.TVLoss_weight=(1, 1)
        else:
            # self.h_x = x.size()[2]
            # self.w_x = x.size()[3]
            # self.batch_size = x.size()[0]
            self.TVLoss_weight = self.compute_weight(weight_map)

        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])

        h_tv = (self.TVLoss_weight[0]*torch.abs((x[:,:,1:,:]-x[:,:,:self.h_x-1,:]))).sum()
        w_tv = (self.TVLoss_weight[1]*torch.abs((x[:,:,:,1:]-x[:,:,:,:self.w_x-1]))).sum()
        # print(self.TVLoss_weight[0],self.TVLoss_weight[1])
        return (h_tv/count_h+w_tv/count_w)/self.batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def compute_weight(self, img):
        gradx = torch.abs(img[:, :, 1:, :] - img[:, :, :self.h_x-1, :])
        grady = torch.abs(img[:, :, :, 1:] - img[:, :, :, :self.w_x-1])
        TVLoss_weight_x = torch.div(1,torch.exp(gradx))
        TVLoss_weight_y = torch.div(1, torch.exp(grady))
        return TVLoss_weight_x, TVLoss_weight_y

def RGB2Y(img):

    r, g, b = torch.split(img, 1, dim=1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y

def norm_1(x):
    max1 = torch.max(x)
    min1 = torch.min(x)
    return (x-min1)/(max1-min1 + 1e-10)

'''     损失      优化器         '''

Mse_loss = nn.MSELoss()
L1_loss = torch.nn.L1Loss()
SL1_loss = torch.nn.SmoothL1Loss()
SSIMLoss = kornia.losses.SSIMLoss(3, reduction='mean')
TDloss = TD(device='cuda')
tv_loss = TVLoss()

optimizer = optim.Adam(model_Dec.parameters(), lr = opt.lr , betas=(0.9, 0.999) )
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [21, 41,61,81], gamma=0.5)

print('---------- Networks initialized -------------')
print('-----------------------------------------------')


loss_plot = []
loss_plot1 = []
loss_plot2 = []
def train(e):
    for iteration, batch in enumerate(training_data_loader, 1):
        imgB,imgA_RGB= batch[0],batch[1]
        imgB = (imgB/255).to(device)
        imgA_RGB = (imgA_RGB/255).to(device)
        imgA = RGB2Y(imgA_RGB)
        _,_,h,w = imgA.shape

        R,ill, IR, Fout, ill_en= model_Dec(imgA,imgB)
        CB = norm_1((imgB - torch.mean(imgB)).clamp_min(1e-10))

        loss1 =  0.5*tv_loss(R) + 10*tv_loss(ill,R)+ F.mse_loss(downsample(ill), downsample(imgA_RGB.max(dim=1, keepdim=True)[0]))
        loss2 = SSIMLoss(imgA,ill*R)+ 2*SSIMLoss(imgB,IR)
        loss3 = 1*(F.mse_loss(Fout*(1-CB) , R.detach()*(1-CB) ) + F.mse_loss(Fout , CB )) + 0.45*TDloss.fusion_TD(R.detach(),Fout)
        loss4 = TDloss(Fout.detach() * ill_en)
        loss = 1* loss1 + loss2  + 2*loss3 + 1e-5*loss4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('This epoch is ' + str(e) , 'loss:', loss.item())
    scheduler1.step()
    if e % 100== 0:
        net_g_model_out_path1 = "./model/model.pth"
        torch.save(model_Dec, net_g_model_out_path1)


if __name__ == '__main__':
    for epoch in range(100):
        train(epoch+1)
        print('this is epoch:' + str(epoch+1) )
    # plt.plot(loss_plot)
    # plt.plot(loss_plot2)
    # plt.show()
