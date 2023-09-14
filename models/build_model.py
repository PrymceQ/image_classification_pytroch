# -*- coding:utf-8 -*-
from torch.hub import load_state_dict_from_url
import torch.nn as nn
import torch.nn.init as init
import torch
import torchvision

from models import LOCAL_PRETRAINED, model_urls
from .efficientnet_pytorch import EfficientNet
from .resnet_pytorch import resnet18, resnet50, resnet101
from .mobilenet_pytorch import MobileNetV2
from .vit_pytorch import ViT

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)

# -----------------------------Model List
def Efficientnet(model_name, num_classes, test=False):
    '''
    model_name :'efficientnet-b2'
    '''
    model = EfficientNet.from_name(model_name)
    if not test:
        if LOCAL_PRETRAINED[model_name] == None:
            state_dict = load_state_dict_from_url(
                model_urls[model_name], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED[model_name])
        model.load_state_dict(state_dict)
    fc_features = model._fc.in_features
    model._fc = nn.Linear(fc_features, num_classes)
    return model

def Resnet(model_name, num_classes, test=False):
    '''
    model_name :'resnet-50'
    '''
    resnet_dict = {
        'resnet-18': resnet18,
        'resnet-50': resnet50,
        'resnet-101': resnet101,
    }
    model = resnet_dict[model_name]()
    if not test:
        if LOCAL_PRETRAINED[model_name] == None:
            state_dict = load_state_dict_from_url(model_urls[model_name], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED[model_name])
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Mobilenet(model_name, num_classes, test=False):
    model = MobileNetV2(width_mult=1.)
    if not test:
        if LOCAL_PRETRAINED[model_name] == None:
            state_dict = load_state_dict_from_url(model_urls[model_name], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED[model_name])
        model.load_state_dict(state_dict)
    fc_features = model.classifier.in_features
    model.classifier = nn.Linear(fc_features, num_classes)
    return model

def VisionTransformer(model_name, num_classes, test=False):
    model = ViT(image_size = 128,
                patch_size = 16,
                num_classes = num_classes,
                dim = 1024,
                depth = 6,
                heads = 16,
                mlp_dim = 2048
                )
    # 对模型权重进行 Xavier 初始化
    initialize_weights(model)

    return model





#
# def Densenet121(num_classes, test=False):
#     model = densenet121()
#     if not test:
#         if LOCAL_PRETRAINED['densenet121'] == None:
#             state_dict = load_state_dict_from_url(model_urls['densenet121'], progress=True)
#         else:
#             state_dict = torch.load(LOCAL_PRETRAINED['densenet121'])
#
#         from collections import OrderedDict
#         new_state_dict = OrderedDict()
#
#         for k,v in state_dict.items():
#             # print(k)  #打印预训练模型的键，发现与网络定义的键有一定的差别，因而需要将键值进行对应的更改，将键值分别对应打印出来就可以看出不同，根据不同进行修改
#             # torchvision中的网络定义，采用了正则表达式，来更改键值，因为这里简单，没有再去构建正则表达式
#             # 直接利用if语句筛选不一致的键
#             ### 修正键值的不对应
#             if k.split('.')[0] == 'features' and (len(k.split('.')))>4:
#                 k = k.split('.')[0]+'.'+k.split('.')[1]+'.'+k.split('.')[2]+'.'+k.split('.')[-3] + k.split('.')[-2] +'.'+k.split('.')[-1]
#             # print(k)
#             else:
#                 pass
#             new_state_dict[k] = v
#         model.load_state_dict(new_state_dict)
#     fc_features = model.classifier.in_features
#     model.classifier = nn.Linear(fc_features, num_classes)
#     return model
#
# def Densenet169(num_classes, test=False):
#     model = densenet169()
#     if not test:
#         if LOCAL_PRETRAINED['densenet169'] == None:
#             state_dict = load_state_dict_from_url(model_urls['densenet169'], progress=True)
#         else:
#             state_dict = torch.load(LOCAL_PRETRAINED['densenet169'])
#
#         from collections import OrderedDict
#         new_state_dict = OrderedDict()
#
#         for k,v in state_dict.items():
#             # print(k)  #打印预训练模型的键，发现与网络定义的键有一定的差别，因而需要将键值进行对应的更改，将键值分别对应打印出来就可以看出不同，根据不同进行修改
#             # torchvision中的网络定义，采用了正则表达式，来更改键值，因为这里简单，没有再去构建正则表达式
#             # 直接利用if语句筛选不一致的键
#             ### 修正键值的不对应
#             if k.split('.')[0] == 'features' and (len(k.split('.')))>4:
#                 k = k.split('.')[0]+'.'+k.split('.')[1]+'.'+k.split('.')[2]+'.'+k.split('.')[-3] + k.split('.')[-2] +'.'+k.split('.')[-1]
#             # print(k)
#             else:
#                 pass
#             new_state_dict[k] = v
#         model.load_state_dict(new_state_dict)
#     fc_features = model.classifier.in_features
#     model.classifier = nn.Linear(fc_features, num_classes)
#     return model


