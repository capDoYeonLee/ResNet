from ast import Pass
import torch
import torch.nn as nn
import torchvision
import numpy 

# function으로 만들지 말고 class로 다시 구축해보자 

def conv_block_1(in_dim, out_dim, act_fn, stride = 1):
    """  bottleneck 구조를 만들기 위한 1x1 convolution """
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 1, stride = stride),
        act_fn
    )
    return model



def conv_block_3(in_dim, out_dim, act_fn):
    """  bottleneck 구조를 만들기 위한 3x3 convolution """
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, stride = 1, padding = 1),
        act_fn
    )
































    






    
    
    