import torch
import torch.nn as nn

from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32
from torchvision.models import swin_t, swin_v2_t

def vit_16_l(im_size,num_class):
    model = vit_l_16(pretrained=False)
    model.heads = nn.Sequential(nn.Linear(1024, num_class))
    return model

def vit_32_l(im_size,num_class):
    model = vit_l_32(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_class)
    return model
def swin_tiny(im_size,num_class):
    model = swin_t(pretrained=False)
    model.head = nn.Linear(768, num_class)
    return model

def swin_tiny2(im_size,num_class):
    model = swin_v2_t(pretrained=False)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img_size = 224
    img = torch.randn(1, 3, img_size, img_size)

    #net = vit_16_l(img_size, 10)
    net = swin_tiny2(img_size, 10)
    out = net(img)
    print(out.shape, count_parameters(net))

