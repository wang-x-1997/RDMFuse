import torch
from torch import nn
import numpy as np
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        # self.con1 = nn.Sequential(
        #     nn.Conv2d(1,16,3,1,1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU()
        # )
        # self.con2 = nn.Sequential(
        #     nn.Conv2d(16, 32, 3, 1, 1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU()
        # )
        # self.con3 = nn.Sequential(
        #     nn.Conv2d(32, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        # self.con4 = nn.Sequential(
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        self.TD1 = nn.Sequential(
            # nn.Conv2d(1, 32, 3, 1, 1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CTNet(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CTNet(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CTNet(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.TD2 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CTNet(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CTNet(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CTNet(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # self.TD2 = CTNet()
        # self.TD3 = CTNet()

        self.ill = illNetwork()

        # self.VI = nn.Sequential(
        #     nn.Conv2d(1,16,3,1,1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, 3, 1, 1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     # nn.Conv2d(256, 256, 3, 1, 1),
        #     # nn.BatchNorm2d(256),
        #     # nn.ReLU()
        # )

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self,x,y):

        # Ef1_1 = self.con1(x)
        # # Ef1_1 = self.TD1(Ef1_1)
        # Ef2_1 = self.con2(Ef1_1)
        # # Ef2_1 = self.TD2(Ef2_1)
        # Ef3_1 = self.con3(Ef2_1)
        # # Ef3_1 = self.TD3(Ef3_1)
        # R = self.con4(Ef3_1)
        # f= self.VI(x)
        R = self.TD1(x)

        ill = self.ill(x)

        # fvi = self.VI(y)
        VI = self.TD2(y)
        # Ef1_2 = self.con1(y)
        # # Ef1_2 = self.TD1(Ef1_2)
        # Ef2_2 = self.con2(Ef1_2)
        # # Ef2_2 = self.TD2(Ef2_2)
        # Ef3_2 = self.con3(Ef2_2)
        # # Ef3_2 = self.TD3(Ef3_2)
        # VI = self.con4(Ef3_2)
        return R, ill, VI


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # self.gls = get_gaussian_kernel()
        # self.dcon1 = nn.Sequential(
        #     nn.Conv2d(256,128,3,1,1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU()
        # )

        # self.dcon2 = nn.Sequential(
        #     nn.Conv2d(128, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        # self.dcon3 = nn.Sequential(
        #     nn.Conv2d(64, 32, 3, 1, 1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU()
        # )
        # self.dcon4 = nn.Sequential(
        #     nn.Conv2d(32, 1, 3, 1, 1),
        #     # nn.BatchNorm2d(128),
        #     nn.Sigmoid()
        # )
        #
        # self.dicon =  nn.Sequential(
        #     nn.Conv2d(128, 64, 3, 1, 1),
        #     nn.Conv2d(64, 32, 3, 1, 1),
        #     nn.Conv2d(32, 1, 3, 1, 1),
        #     nn.Sigmoid()
        # )
        self.VI1 = nn.Sequential(
            # nn.Conv2d(64, 32, 3, 1, 1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Conv2d(32, 32, 3, 1, 1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            # nn.BatchNorm2d(128),
            nn.Sigmoid()
        )
        # self.VI2 = nn.Sequential(
        #     nn.Conv2d(64, 32, 3, 1, 1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     # nn.Conv2d(64, 32, 3, 1, 1),
        #     # nn.BatchNorm2d(32),
        #     # nn.ReLU(),
        #     nn.Conv2d(32, 1, 3, 1, 1),
        #     # nn.BatchNorm2d(128),
        #     nn.Sigmoid()
        # )
        self.VI = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            # nn.BatchNorm2d(128),
            nn.Sigmoid()
        )
        self.VI2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            # nn.BatchNorm2d(128),
            nn.Sigmoid()
        )
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)
    def forward(self, x,y,IR):

        R = self.VI(x)

        ill = self.VI1(y)
        IR = self.VI2(IR)
        # ill = self.gls(ill)

        # D1 = self.dcon2(IR)
        # D1 = self.dcon3(D1)
        # IR = self.VI2(IR)

        return R, ill,IR

class DecNet(nn.Module):

    def __init__(self):
        super(DecNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.enhance = Enhance()
        self.Fusionde = FusionNet()

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self,VIS,IR):
        f_R,f_ill,f_IR = self.encoder(VIS,IR)

        R,ill,IR = self.decoder(f_R,f_ill,f_IR)

        Fout = self.Fusionde(f_R.detach(),f_IR.detach())
        ill_en = self.enhance(ill.detach().clamp(min=0,max=1),Fout.detach())

        return R,ill, IR, Fout, ill_en

# class Fusion_en(nn.Module):
#     def __init__(self):
#         super(Fusion_en, self).__init__()
#         self.enhance = Enhance()
#         self.Fusionde = FusionNet()
#
#     def weights_init(self, m):
#         if isinstance(m, nn.Conv2d):
#             m.weight.data.normal_(0, 0.02)
#             m.bias.data.zero_()
#
#         if isinstance(m, nn.BatchNorm2d):
#             m.weight.data.normal_(1., 0.02)
#
#     def forward(self,VI, f_R,f_IR ,ill):
#
#         F = self.Fusionde(f_R.detach(),f_IR.detach())
#
#         ill_1 = ill.detach()
#         # R_1 = R.detach()
#         ill_en = self.enhance(ill_1,VI)
#         ill_en = ill_en.clamp_min(1e-10)
#         return F,ill_en




class illNetwork(nn.Module):
    def __init__(self, ):
        super(illNetwork, self).__init__()
        layers = 2
        channels = 16

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        # self.out_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1),
        #     # nn.Sigmoid()
        # )
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)
    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = conv(fea)+fea
        # fea = self.out_conv(fea)

        # illu = fea + input
        # illu = torch.clamp(fea, 0.0001, 1)

        return fea

class Enhance(nn.Module):
    def __init__(self):
        super(Enhance, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2,16,3,1,1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(32,1,3,1,1),
            nn.Sigmoid()
        )

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)
    def forward(self,x,y):
        # outset = x
        output = self.conv(torch.cat([x,y],1))
        avg = self.avg(output)*x
        max = self.max(output)*x
        # xx = self.conv1(torch.cat([avg, max], 1))
        output = self.conv1(torch.cat([avg,max],1))+x

        return output

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(128,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,16,3,1,1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,1,3,1,1),
            nn.Sigmoid()
        )

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)
    def forward(self,IR,VI):
        F = self.conv(torch.cat([IR,VI],1))
        return F


class CTNet(nn.Module): #contrast and texture
    def __init__(self,inchanel):
        super(CTNet, self).__init__()
        # self.fusion = FusionNet()
        self.con = nn.Conv2d(1,inchanel,1,1,0)
        self.con1 = nn.Conv2d(inchanel, inchanel, 3, 1, 1)
        self.con2 = nn.Conv2d(inchanel*1,inchanel,1,1,0)
        self.channel= inchanel

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)
    def forward(self,x):

        T = self.con(Sobel(x,channel=self.channel) )
        C = norm_1((x - torch.mean(x).clamp_min(1e-10)))
        R = self.con2(x+C+T) + x
        # C = torch.cat([C,x],1)
        # R = self.con1(x) + x
        return R




def norm_1(x):
    max1 = torch.max(x)
    min1 = torch.min(x)
    return (x-min1)/(max1-min1 + 1e-10)

def Sobel(image, channel,cuda_visible=True):
    assert torch.is_tensor(image) is True
    # [-1.,-2.,-1.], [0.,0.,0.,], [1.,2.,1.]
    # [0., 1., 0.], [1., -4., 1.], [0., 1., 0.]
    # [-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1]
    laplace_operator = np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=np.float32)[np.newaxis,
                       :, :].repeat(channel, 0)
    if cuda_visible:
        laplace_operator = torch.from_numpy(laplace_operator).unsqueeze(0).cuda()
    else:
        laplace_operator = torch.from_numpy(laplace_operator).unsqueeze(0)

    image = F.conv2d(image, laplace_operator,padding=1, stride=1)

    return image

import math
def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


