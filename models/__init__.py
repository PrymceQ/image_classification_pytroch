# __all__ = ['LOCAL_PRETRAINED', 'model_urls', 'Efficientnet', 'Resnet', 'Mobilenet']

## 预训练模型的存放位置
LOCAL_PRETRAINED = {
    'resnet-50': 'ckpts/pretrained_resnet50.pth',
    'mobilenet-v2': 'ckpts/pretrained_mobilenet-v2.pth',
    'efficientnet-b2': 'ckpts/pretrained_efficientnet-b2.pth',
    'vit': '',
}

model_urls = {
    # resnet
    'resnet-18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet-34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet-50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet-101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet-152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',

    'densenet-121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet-169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    # mobilenet
    'mobilenet-v2': 'https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_1.0-0c6065bc.pth',
    # efficientnet
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',
}

# from .vision import *
from .build_model import *