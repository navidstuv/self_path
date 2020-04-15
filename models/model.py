"""
Multi-Task Histology model
"""
import torch.nn as nn
from .encoder import get_resnet, Gen128
from .decoders import  UnetDecoder, Classifier, Disc128_classifier
from torch import optim


class MultiTaskCNN(nn.Module):
    """
    Multi-Task survival CNN model with attention block
    """
    def __init__(self, encoder_name='resnet50', pretrained=False):
        """
        """
        super(MultiTaskCNN, self).__init__()
        self.base, self.latent_dim  = get_resnet(encoder_name, pretrained=pretrained)
        self.decoders = nn.ModuleDict({})

    def forward(self, x, task_name, req_inter_layer=False):
        """
        Forward pass through the model
        :param x: input features
        :param task_id: task index number
        :return:
        """
        x = self.base(x)

        if isinstance(self.decoders[task_name], UnetDecoder):
            out = self.decoders[task_name](x, layer0, layer1, layer2, layer3, layer4)
            return out
        if isinstance(self.decoders[task_name], Disc128_classifier):
            if req_inter_layer==True:
                out, inter_layer = self.decoders[task_name](x, req_inter_layer)
                return out, inter_layer
            else:
                out = self.decoders[task_name](x, req_inter_layer)
                return out



def get_model(config):
    """
    Get model initialised
    :param config:
    :return:
    """
    d_model = MultiTaskCNN(encoder_name=config.encoder_name, pretrained=config.pretrained).apply(init_normal)
    g_model = Gen128(latent_dim=100).apply(init_normal)
    latent_dim = d_model.latent_dim
    for task_name in config.task_names:
        task_dictionary = config.tasks[task_name]
        task_type = task_dictionary['type']
        n_classes = task_dictionary['n_classes']

        if task_type == 'segmentation':
            d_model.decoders.update({task_name:UnetDecoder(n_classes,
                                                           d_model.base.multiple)})
        if task_type == 'classification':
            d_model.decoders.update({task_name:
                                       Disc128_classifier(input_dim=latent_dim, n_classes=n_classes)})
    # Place model on cuda
    d_model = d_model.cuda()
    g_model = g_model.cuda()
    return d_model, g_model

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


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0.0, std=.05)
        nn.init.constant_(m.bias, 0.0)

    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, mean=0.0, std=.05)

    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, mean=0.0, std=.05)
        nn.init.constant_(m.bias, 0.0)

    if hasattr(m, 'weight_g'):
        nn.init.constant_(m.weight_g,1)