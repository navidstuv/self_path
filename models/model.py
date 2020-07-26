"""
Multi-Task Histology model
"""
import torch.nn as nn
from .encoder import get_resnet, ResNet
from .decoders import  UnetDecoder, Classifier, Disc128_classifier, Classifier_wide_resnet
from torch import optim
from .wide_resnet import Wide_ResNet
from utils.utils import ReverseLayerF


class MultiTaskCNN(nn.Module):
    """
    Multi-Task survival CNN model with attention block
    """
    def __init__(self, encoder_name='resnet50', pretrained=False):
        """
        """
        super(MultiTaskCNN, self).__init__()
        # self.base, self.latent_dim  = get_resnet(encoder_name, pretrained=pretrained)
        # self.decoders = nn.ModuleDict({})
        self.encoder_name = encoder_name
        self.bn = nn.BatchNorm2d(3)
        if self.encoder_name == 'wide_resnet':
            self.base = Wide_ResNet(28, 2, 0.3, 10)
        if self.encoder_name == 'resnet50':
            self.base = ResNet(encoder_name, pretrained=pretrained)
        self.decoders = nn.ModuleDict({})

    def forward(self, x, task_name, alpha=1):
        """
        Forward pass through the model
        :param x: input features
        :param task_id: task index number
        :return:
        """
        if self.encoder_name == 'resnet50':
            x = self.bn(x)
            layer0, layer1, layer2, layer3, layer4 = self.base(x)
            if task_name=='domain_classifier':
                if isinstance(self.decoders[task_name], Disc128_classifier):
                    reversed_input = ReverseLayerF.apply(layer4, alpha)
                    out = self.decoders[task_name](reversed_input)
            else:
                if isinstance(self.decoders[task_name], UnetDecoder):
                    out = self.decoders[task_name](x, layer0, layer1, layer2, layer3, layer4)
                if isinstance(self.decoders[task_name], Classifier):
                    out = self.decoders[task_name](layer4)
            return out
        if self.encoder_name == 'wide_resnet':
            encoder_out = self.base(x)
            if task_name=='domain_classifier':
                if isinstance(self.decoders[task_name], Classifier_wide_resnet):
                    reversed_input = ReverseLayerF.apply(layer4, alpha)
                    out = self.decoders[task_name](reversed_input)
            else:
                if isinstance(self.decoders[task_name], UnetDecoder):
                    out = self.decoders[task_name](x, layer0, layer1, layer2, layer3, layer4)
                if isinstance(self.decoders[task_name], Classifier_wide_resnet):
                    out = self.decoders[task_name](encoder_out)
            return out





def get_model(config):
    """
    Get model initialised
    :param config:
    :return:
    """
    model = MultiTaskCNN(encoder_name=config.encoder_name, pretrained=config.pretrained)
    if config.encoder_name =='resnet50':
        latent_dim = model.base.latent_dim
    for task_name in config.task_names:
        task_dictionary = config.tasks[task_name]
        task_type = task_dictionary['type']
        n_classes = task_dictionary['n_classes']
        if task_type == 'pixel_self':
            model.decoders.update({task_name:UnetDecoder(n_classes,
                                            model.base.multiple)})
        if task_type in ['classification_adapt', 'classification_self', 'classification_main']:
            if config.encoder_name == 'wide_resnet':
                model.decoders.update({task_name:
                                           Classifier_wide_resnet(input_dim=2048, n_classes=n_classes)})
            if  config.encoder_name == 'resnet50':
                model.decoders.update({task_name:
                                  Classifier(input_dim=latent_dim, n_classes=n_classes)})
    # Place model on cuda
    model = model.cuda()
    return model