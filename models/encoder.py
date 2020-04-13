"""
Implementation of dilated ResNet-101 with deep supervision. Downsampling is changed to 8x
"""
import numpy as np
from torch import nn
import torchvision.models as models
import torch
from torch.nn import BatchNorm2d


def get_resnet(encoder_name, pretrained=True):
    """
    Returns a resnet18 or resnet50 model, with AdaptiveAvgPool2d
    :param adaptive_pool_dim: dimension of feature output from adaptive
        average pooling h * w
    :param pretrained: if the model should be Imagenet pretrained
    :return: model, cnn_features_dim
    """

    if encoder_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        latent_dim = 512
    else:
        model = models.resnet50(pretrained=pretrained)
        latent_dim = 2048

    children = (
        [BatchNorm2d(3)]
        + list(model.children())[:-2])
    model = torch.nn.Sequential(*children)
    return model, latent_dim


class ResNet(nn.Module):

    def __init__(self, encoder_name, pretrained, detach=False):
        super(ResNet, self).__init__()

        self.detach = detach

        if encoder_name == "resnet18":
            self.base_model = models.resnet18(pretrained=pretrained)
            self.latent_dim = 512
            self.multiple = 1
        else:
            self.base_model = models.resnet50(pretrained=pretrained)
            self.latent_dim = 2048
            self.multiple = 4

        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        detach = int(self.detach) if isinstance(self.detach, bool) else np.random.binomial(1, self.detach)
        if detach == 1:
            return layer0.detach(), layer1.detach(), layer2.detach(), layer3.detach(), layer4
        if detach == 0:
            return layer0, layer1, layer2, layer3, layer4