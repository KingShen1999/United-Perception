#!/usr/bin/env python
# coding: utf-8
# %%
import os
import torch
import torch.nn as nn
from torchstat import stat
import math
import time
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# %%
def channel_split(x, split):
    return torch.split(x, split, dim=1)


# %%
class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.data.size(0),-1)


# %%
class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1',ConvBnAct(in_channels, out_channels, kernel_size = 1))
        self.add_module('layer2',DWConvLayer(out_channels, out_channels,stride=stride))
    def forward(self, x):
        return super().forward(x)


# %%
class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels,  stride=1,  bias=False):
        super().__init__()
        out_ch = out_channels
        groups = in_channels
        kernel = 3
        self.add_module('dwconv', nn.Conv2d(groups, groups, kernel_size=3,
                                    stride=stride, padding=1, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(groups))
    def forward(self, x):
        return super().forward(x) 


# %%
class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,kernel_size//2,dilation,groups,bias)
        self.bn=nn.BatchNorm2d(out_channels)
        if apply_act:
            self.act=nn.ReLU6(inplace = True)
        else:
            self.act=nn.Identity()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# %%
class SEModule(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""
    def __init__(self, w_in, reduction_rate = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1=nn.Conv2d(w_in, w_in//reduction_rate, 1, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(w_in//reduction_rate, w_in, 1, bias=True)
        self.act2=nn.Sigmoid()

    def forward(self, x):
        y=self.avg_pool(x)
        y=self.act1(self.conv1(y))
        y=self.act2(self.conv2(y))
        return x * y


# %%
class CSPKingBlock(nn.Module):
    def get_out_ch(self):
        return self.kingouch+self.patial_channel
    
    def __init__(self, in_channels,n_layers,patial_channel,dwconv=False):
        super().__init__()
        self.patial_channel = patial_channel
        self.KingBlk = KingBlock(in_channels-patial_channel,n_layers ,dwconv=dwconv)
        ouch = self.KingBlk.get_out_ch()
        self.kingouch = ouch
        self.transition = nn.Sequential(#SELayer(ouch),
                                       ConvBnAct(ouch, ouch, kernel_size=1))
    def forward(self, x):
        x1 = x[:,:self.patial_channel,:,:]   #cross stage
        x2 = x[:,self.patial_channel:,:,:]
        x2 = self.KingBlk(x2)
        x2 = self.transition(x2)
        x = torch.cat([x1,x2],dim = 1)
        return x


# %%

# %%
class KingBlock(nn.Module):
    def get_divisor(self): #get the divisors of n
        divisors = []
        for i in range(1,self.n_layers+1):
            if(int(self.n_layers/i)==self.n_layers/i):
                divisors.append(i)
        return divisors
    def get_link(self):  #calculate the linkcount of all layer
        links = [[] for x in range(self.n_layers)]
        for div in self.divisors:
            for k in range(0,self.n_layers,div):
                links[k].append(div)
        return links

    def get_out_ch(self):
        link_count = 0
        for out in self.concate_out[self.n_layers]:
            link_count+=len(self.links[out])
        return self.growth*link_count+self.in_channels
            
    def __init__(self, in_channels,  n_layers, dwconv=False):
        super().__init__()
        self.n_layers=n_layers
        self.in_channels = in_channels
        self.divisors = self.get_divisor()
        self.links = self.get_link()
        self.concate_out={3:[1],4:[1,3],6:[2,4],8:[1,3,5,7],9:[3,6],10:[2,5,8],12:[3,6,9],15:[3,6,9,12],16:[2,6,10,14]}
        self.growth = int(self.in_channels/len(self.divisors))
        layers_ = []
        for i in range(n_layers):
            if(i!=n_layers-1):
                channel = len(self.links[i+1])*self.growth
            else:
                channel = self.in_channels
            if dwconv:
                if(i+1 in self.concate_out[self.n_layers]):
                    layers_.append(CombConvLayer(channel, channel))
                else:
                    layers_.append(CombConvLayer(channel, channel))
                
            else:
                if(i+1 in self.concate_out[self.n_layers]):
                    layers_.append(ConvBnAct(channel, channel))
                else:
                    layers_.append(ConvBnAct(channel, channel))
        self.layers = nn.ModuleList(layers_)
        
    def forward(self, x):
        layers_ = [x]
        tensors = [[] for i in range(self.n_layers)]
        for layer in range(len(self.layers)):
            tins = channel_split(x,self.growth)
            for i in range(len(tins)):
                tensors[layer+self.links[layer][i]-1].append(tins[i])
            if len(tensors[layer])>1:
                x = torch.cat(tensors[layer], dim=1)
            else:
                x = tensors[layer][0]
            x = self.layers[layer](x)
            
            layers_.append(x)
        
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == t-1) or (i in self.concate_out[self.n_layers]):
                out_.append(layers_[i])
                
        out = torch.cat(out_, 1)
        #----------不用改----------
        return out


# %%
class KingNet(nn.Module):
    def __init__(self, in_channels=3,arch=53 ,depth_wise=False,pretrained=False):
        super().__init__()
        second_kernel = 3
        
        if(arch==20):
            first_ch  = [8, 16]
            ch_list = [32, 64, 128]
            n_layers = [8, 8, 8]
            downSamp = [   1 ,  1,  0]
            drop_rate = 0.05
            self.out_strides = [4,8,16]
            self.out_planes = ch_list
        if(arch==24):
            first_ch  = [6, 12]
            ch_list = [24, 48, 96, 192]
            n_layers = [4, 4, 4, 4]
            downSamp = [   1 ,  1,  1,  0]
            drop_rate = 0.05
            self.out_strides = [4,8,16,32]
            self.out_planes = ch_list
        if(arch==42):
            first_ch  = [24, 48]
            ch_list = [96, 192, 384, 768]
            n_layers = [8, 8, 8, 8]
            downSamp = [   1 ,  1,  1,  0]
            drop_rate = 0.05
            self.out_strides = [4,8,16,32]
            self.out_planes = ch_list
        if(arch==53):
            first_ch  = [30,60]
            ch_list = [120, 240, 540, 800,1200]
            partial_channel = [15,30,60,135,200]
            n_layers = [ 9, 9, 15, 9,3]
            downSamp = [   1 ,  0,  1,  1,  0]
            drop_rate=0.1
            self.out_strides = [4,8,16,32]
            self.out_planes = [120,  540, 800,1200]
        if arch==69:
            first_ch  = [48, 96]
            ch_list = [ 160, 240, 400, 720, 1080, 1440]
            partial_channel = [24,40,60,100,180,270]
            n_layers = [   9,  9,  15,  15,  9,   3]
            downSamp = [   1,   0,   1,   0,   1,   0]
            drop_rate = 0.2
            
        max_pool = True
        if depth_wise:
            second_kernel = 1
            max_pool = False
            drop_rate = 0.05
         
        blks = len(n_layers)
        self.base = nn.ModuleList([])
        

        self.base.append(ConvBnAct(in_channels,first_ch[0],3,2,1))
        self.base.append(ConvBnAct(first_ch[0],first_ch[1],3,2,1))


            
        # Build all KingNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = KingBlock(ch,n_layers[i],dwconv=depth_wise)
            ouch = blk.get_out_ch()
            self.base.append (blk)
                
            ch=ch_list[i]
            if downSamp[i] == 1:
                self.base.append(ConvBnAct(ouch,ch_list[i],3,2,1))
                
        
        ch = ch_list[blks-1]

    def forward(self, input):
        x = input['image']
        features = []
        for i in range(len(self.base)):
            x = self.base[i](x)
            #print(i)
            #print(self.base[i])
            if i==3 or i==5 or i==6:
                features.append(x)
        
        return {'features': features, 'strides': self.get_outstrides()}
        
    def get_outplanes(self):
        return self.out_planes

    def get_outstrides(self):
        return torch.tensor(self.out_strides, dtype=torch.int)

# %%

# %%
net = KingNet(arch=20)
#stat(net,(3,96,96))
#x = torch.rand(1,3,224,224)
#x = net.forward(x)
#print(x)

# %%
def kingnet42(pretrained=False, **kwargs):
    """
    Constructs a KingNet-42 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = KingNet(arch=42)
    #if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


# %%
def kingnet24(pretrained=False, **kwargs):
    """
    Constructs a KingNet-42 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = KingNet(arch=24)
    #if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


# %%
def kingnet20(pretrained=False, **kwargs):
    """
    Constructs a KingNet-42 model.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = KingNet(arch=20)
    #if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

# %%
#net = KingNet(arch=53)


# %%
def speed(network):
    if(network=="hardnet"):
        model = HarDNet(arch=85,depth_wise=False)
    else:
        model = KingNet(arch=24)
    total_params = sum(p.numel() for p in model.parameters())
    #print('Parameters: ', total_params )
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()
    model.to(device)
    total_time = 0
    start_time = 0
    time_all = 0    
    #images = tor
    for i in range(100):
        images = torch.randn((1, 3, 112, 112)).cuda()
        data = {"image":images}
        if i == 0:
            with torch.no_grad():
                output = model(data)
        else:
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                output = model(data)
            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - start_time


            print(
                "Inference time \
                  (iter {0:5d}):  {1:3.5f} fps".format(
                    i + 1, 1 / elapsed_time
                )
            )
            total_time += 1/elapsed_time
    print(total_time/100)

# %%
speed("kingnet")

# %%

# %%

# %%
