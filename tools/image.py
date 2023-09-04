import torch
import argparse
from PIL import Image

from config.config import get_cfg
from data import Compose, ColorStyle, Resize, ToTensor, Normalize

def set_cfg(yaml_file):
    cfg = get_cfg()
    cfg.merge_from_file(yaml_file)
    return cfg

def load_checkpoint(weight_path):
    checkpoint = torch.load(weight_path)
    model = checkpoint['model']  # 提取网络结构
    # model.set_swish(memory_efficient=False)
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model

def load_img(img_path, transform):
    img = Image.open(img_path).convert('RGB')
    img, _ = transform(img=img, label=None)

    if torch.cuda.is_available():
        img = img.unsqueeze(0).cuda()

    return img

def inference(model, img):
    with torch.no_grad():
        out = model(img)
    out = torch.softmax(out, dim=1)

    out_scores = out.cpu().numpy()[0]
    pre_score = torch.max(out).cpu().item()
    pre_index = torch.argmax(out, dim=1).cpu().item()
    return out_scores, pre_score, pre_index

if __name__=="__main__":
    parser = argparse.ArgumentParser('Classification inference code!!')
    parser.add_argument('--img', type=str, default='dataset/test/1/00000_00000.jpg', help='image path')
    parser.add_argument('--cfg', type=str, default='config/default_config.yaml', help='config path')
    parser.add_argument('--weight', type=str, default='weights/test/best.pth', help='weight(.pth) path')
    # parser.add_argument('--thre', type=float, default=0.83)
    opt = parser.parse_args()

    # Load cfg
    cfg = set_cfg(opt.cfg)

    # Load model
    model = load_checkpoint(opt.weight)
    print('..... Finished loading model!......')

    # Load image
    infer_transform = Compose([
        ColorStyle(cfg.TEST.COLOR_STYLE),
        Resize(cfg.TEST.INPUT_SIZE),
        ToTensor(),
        Normalize(mean=cfg.TEST.MEAN, std=cfg.TEST.STD),
    ])
    img = load_img(opt.img, transform=infer_transform)
    print('..... Finished loading image!......')

    # Inference
    out_scores, pre_score, pre_index = inference(model, img)
    pre_classname = cfg.DATASET.IDX_MAP[pre_index]

    print('Predict scores: {} \nPre_score: {:.5f} \nPre_index: {} \nPre_className: {}'.format(out_scores, pre_score, pre_index, pre_classname))



