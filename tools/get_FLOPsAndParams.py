import torch
from thop import profile

import argparse
from torchstat import stat
import warnings


def get_args():
    parser = argparse.ArgumentParser('Convert pth -> onnx')
    parser.add_argument('--model_path', type=str, default='weights/test/best.pth', help='model_path')
    parser.add_argument('--img_size', type=int, default=512)
    args = parser.parse_args()
    return args


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


if __name__ == '__main__':
    opt = get_args()

    # Load model
    model = load_checkpoint(opt.model_path)
    if torch.cuda.is_available():
        model.cuda()
    print('..... Finished loading model! ......')

    # Get input
    dummy_input = torch.randn([1, 3, opt.img_size, opt.img_size])
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    print('..... Finished creating input! ......')

    # Get params(M) and FLOPs(GFLOPs)
    flops, params = profile(model, inputs=(dummy_input,))
    params_in_million = round(params / 1_000_000, 4)
    gflops = round(params / 1e9, 6)

    print('\n..... INFOs!!! ......')
    print(f"Total parameters: {params_in_million} M")
    print(f"Total FLOPs: {gflops} GFLOPs")