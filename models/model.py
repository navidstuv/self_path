"""
Multi-Task Histology model
"""
import torch.nn as nn
from .encoder import get_resnet
from .decoders import  UnetDecoder, Classifier
from torch import optim
from pytorch_revgrad import RevGrad
from utils.utils import ReverseLayerF


class MultiTaskCNN(nn.Module):
    """
    Multi-Task survival CNN model with attention block
    """
    def __init__(self, encoder_name='resnet50', pretrained=False):
        """
        """
        super(MultiTaskCNN, self).__init__()
        self.base, self.latent_dim  = get_resnet(encoder_name, pretrained=pretrained)
        self.reverse_grad = RevGrad()
        self.decoders = nn.ModuleDict({})

    def forward(self, x, task_name, alpha=1):
        """
        Forward pass through the model
        :param x: input features
        :param task_id: task index number
        :return:
        """
        x = self.base(x)
        if task_name=='domain_classifier':
            if isinstance(self.decoders[task_name], UnetDecoder):
                out = self.decoders[task_name](x, layer0, layer1, layer2, layer3, layer4)
            if isinstance(self.decoders[task_name], Classifier):
                reversed_input = ReverseLayerF.apply(x, alpha)
                out = self.decoders[task_name](reversed_input)
        else:
            if isinstance(self.decoders[task_name], UnetDecoder):
                out = self.decoders[task_name](x, layer0, layer1, layer2, layer3, layer4)
            if isinstance(self.decoders[task_name], Classifier):
                out = self.decoders[task_name](x)
        return out




def get_model(config):
    """
    Get model initialised
    :param config:
    :return:
    """
    model = MultiTaskCNN(encoder_name=config.encoder_name, pretrained=config.pretrained)
    latent_dim = model.latent_dim
    for task_name in config.task_names:
        task_dictionary = config.tasks[task_name]
        task_type = task_dictionary['type']
        n_classes = task_dictionary['n_classes']

        if task_type == 'segmentation':
            model.decoders.update({task_name:UnetDecoder(n_classes,
                                            model.base.multiple)})
        if task_type == 'classification':
            model.decoders.update({task_name:
                                  Classifier(input_dim=latent_dim, n_classes=n_classes)})
    # Place model on cuda
    model = model.cuda()
    return model

def get_optimizer(model, optimizer_type, lr, weight_decay):
    """
    :param model: pytorch model
    :param optimizer_type: SGD or ADAM
    :param lr:
    :param weight_decay:
    :return:
    """
    if optimizer_type == 'SGD':
        return optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    if optimizer_type == 'ADAM':
        return optim.Adam(model.parameters(), lr=lr,
                          weight_decay=weight_decay)
    else:
        raise Exception('Wrong optimizer type {}'.format(optimizer_type))