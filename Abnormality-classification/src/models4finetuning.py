'''
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import collections as co
from torch.autograd import Variable
import torchvision.models as models
def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    if model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Sequential(nn.Conv2d(512, 512, kernel_size=(1,1), stride=(1,1)),
                                                nn.Dropout(p=0.8),
                                                nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1)))
        model_ft.num_classes = num_classes
    
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft
