# 肝脏分割论文网络模型
# 2023-04-11 重新编改

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.convolutions import Convolution


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1=Convolution(spatial_dims=2,in_channels=in_channels,out_channels=out_channels)
        self.conv2 = Convolution(spatial_dims=2, in_channels=out_channels, out_channels=out_channels)
    def forward(self, x):
        y=self.conv1(x)
        return self.conv2(y)+y


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1=Convolution(spatial_dims=2,in_channels=in_channels,out_channels=out_channels)
        self.conv2 = Convolution(spatial_dims=2, in_channels=out_channels, out_channels=out_channels)
    def forward(self, x):
        y=self.conv1(x)
        return self.conv2(y)

class Down(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x):
        return self.up(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class DAS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DAS, self).__init__()
        self.conv1 = Convolution(spatial_dims=2,in_channels=in_channels,out_channels=in_channels,kernel_size=3)
        self.conv2 = Convolution(spatial_dims=2, in_channels=in_channels, out_channels=in_channels, kernel_size=3)
        self.conv3 = Convolution(spatial_dims=2, in_channels=in_channels, out_channels=in_channels, kernel_size=3)

        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        px = self.conv1(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        py = self.conv2(x).view(m_batchsize, -1, width * height)
        pz = self.conv3(x).view(m_batchsize, -1, width * height)

        pxy=self.sigmoid(torch.bmm(px,py))

        pxyz=torch.bmm(pz,pxy).view(m_batchsize, -1, height, width)+x

        cy=x.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        cx=x.view(m_batchsize, -1, width * height)
        cz = x.view(m_batchsize, -1, width * height)
        cxy = self.sigmoid(torch.bmm(cx,cy))
        cxyz = torch.bmm(cxy,cz).view(m_batchsize, -1, height, width)+x

        out = pxyz + cxyz + x
        return out


class DAS_UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(DAS_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.resblock1=ResBlock(in_channels=self.n_channels,out_channels=64)
        self.down1=Down()
        self.resblock2=ResBlock(in_channels=64,out_channels=128)
        self.down2=Down()
        self.resblock3=ResBlock(in_channels=128,out_channels=256)
        self.down3=Down()
        self.resblock4=ResBlock(in_channels=256,out_channels=512)

        self.das=DAS(512,512)

        self.up1=Up()
        self.convblock1=ConvBlock(256+512,256)
        self.up2=Up()
        self.convblock2=ConvBlock(128+256,128)
        self.up3=Up()
        self.convblock3=ConvBlock(64+128,64)

        self.outcov=OutConv(64,self.n_classes)



    def forward(self,x):
        y1 = self.resblock1(x)
        y2 = self.resblock2(self.down1(y1))
        y3 = self.resblock3(self.down2(y2))
        y4 = self.resblock4(self.down3(y3))
        y4 = self.das(y4)
        u1 = self.convblock1(torch.cat([y3,self.up1(y4)],dim=1))
        u2 = self.convblock2(torch.cat([y2, self.up2(u1)], dim=1))
        u3 = self.convblock3(torch.cat([y1, self.up3(u2)], dim=1))

        out=self.outcov(u3)

        return out


