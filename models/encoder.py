"""
Implementation of dilated ResNet-101 with deep supervision. Downsampling is changed to 8x
"""
import numpy as np
from torch import nn
import torchvision.models as models
import torch
from torch.nn import BatchNorm2d
from torch.nn.utils import weight_norm


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
        children = (
                [BatchNorm2d(3)]
                + list(model.children())[:-2])
        model = torch.nn.Sequential(*children)
    elif encoder_name == "resner50":
        model = models.resnet50(pretrained=pretrained)
        latent_dim = 2048
        children = (
                [BatchNorm2d(3)]
                + list(model.children())[:-2])
        model = torch.nn.Sequential(*children)
    elif encoder_name == "resnet101":
        model = models.resnet50(pretrained=pretrained)
        latent_dim = 2048
        children = (
                [BatchNorm2d(3)]
                + list(model.children())[:-2])
        model = torch.nn.Sequential(*children)

    elif encoder_name =="Disc128":
        model = Disc128()
        latent_dim = 192
    return model, latent_dim


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(len(x), -1)

class Disc128(nn.Module):
    """docstring for Discriminator"""

    def __init__(self):
        super(Disc128, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(.2),
            weight_norm(nn.Conv2d(3, 96, 3, stride=1, padding=1)),
            nn.LeakyReLU(.2),
            weight_norm(nn.Conv2d(96, 96, 3, stride=1, padding=1)),
            nn.LeakyReLU(.2),
            weight_norm(nn.Conv2d(96, 96, 3, stride=2, padding=1)),
            nn.LeakyReLU(.2),

            nn.Dropout(.5),
            weight_norm(nn.Conv2d(96, 128, 3, stride=1, padding=1)),
            nn.LeakyReLU(.2),
            weight_norm(nn.Conv2d(128, 128, 3, stride=1, padding=1)),
            nn.LeakyReLU(.2),
            weight_norm(nn.Conv2d(128, 128, 3, stride=2, padding=1)),
            nn.LeakyReLU(.2),

            nn.Dropout(.5),
            weight_norm(nn.Conv2d(128, 192, 3, stride=1, padding=1)),
            nn.LeakyReLU(.2),
            weight_norm(nn.Conv2d(192, 192, 3, stride=1, padding=1)),
            nn.LeakyReLU(.2),
            weight_norm(nn.Conv2d(192, 192, 3, stride=2, padding=1)),
            nn.LeakyReLU(.2),

            nn.Dropout(.5),
            weight_norm(nn.Conv2d(192, 192, 3, stride=1, padding=0)),
            nn.LeakyReLU(.2),
            weight_norm(nn.Conv2d(192, 192, 1, stride=1, padding=0)),
            nn.LeakyReLU(.2),
            weight_norm(nn.Conv2d(192, 192, 1, stride=1, padding=0)),
            nn.LeakyReLU(.2),

            # nn.AvgPool2d(6,stride=1),
            # nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
            Flatten()
        )

    def forward(self, x):
        inter_layer = self.net(x)
        return inter_layer

class Reshape(nn.Module):

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.req_shape = (-1,)+shape

    def forward(self, x):
        return x.view(self.req_shape)
class Gen128(nn.Module):
    """docstring for Generator"""

    def __init__(self, latent_dim):
        super(Gen128, self).__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, 512*4*4),
            nn.BatchNorm1d(512*4*4),
            nn.ReLU(),
            Reshape((512, 4, 4)),
            nn.ConvTranspose2d(512, 256, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            weight_norm(nn.ConvTranspose2d(128, 3, 5, stride=2, padding=2, output_padding=1)),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


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