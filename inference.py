import os
import argparse
from tqdm.contrib import tzip

from tools.image import set_cfg, load_checkpoint, load_img, inference
from data import Compose, ColorStyle, Resize, ToTensor, Normalize

if __name__=="__main__":
    parser = argparse.ArgumentParser('Classification inference code!!')
    parser.add_argument('--img-folder', type=str, default='dataset/test/1', help='image folder path')
    parser.add_argument('--cfg', type=str, default='config/default_config.yaml', help='config path')
    parser.add_argument('--weight', type=str, default='weights/test/best.pth', help='weight(.pth) path')
    parser.add_argument('--out', type=str, default='out.txt', help='predict result output file (.txt)')
    opt = parser.parse_args()

    # Load cfg
    cfg = set_cfg(opt.cfg)
    CLASS_NAME = cfg.DATASET.IDX_MAP

    # Load model
    model = load_checkpoint(opt.weight)
    print('..... Finished loading model!......')

    # Load images
    infer_transform = Compose([
        ColorStyle(cfg.TEST.COLOR_STYLE),
        Resize(cfg.TEST.INPUT_SIZE),
        ToTensor(),
        Normalize(mean=cfg.TEST.MEAN, std=cfg.TEST.STD),
    ])
    imgs_path = [os.path.join(opt.img_folder, i) for i in os.listdir(opt.img_folder) if i.endswith('jpg')]    # file path
    imgs = [load_img(i, transform=infer_transform) for i in imgs_path]  # img torch.Tensor
    print('..... Finished loading {} images!......'.format(len(imgs)))

    # Inference
    infos = []
    for ip, im in tzip(imgs_path, imgs):
        out_scores, pre_score, pre_index = inference(model, im)
        pre_classname = cfg.DATASET.IDX_MAP[pre_index]
        infos.append((ip, list(out_scores), pre_score, pre_index, pre_classname))

        print('Img: {}\n'.format(ip))
        print('Predict scores: {} \nPre_score: {:.5f} \nPre_index: {} \nPre_className: {}'.format(out_scores, pre_score, pre_index, pre_classname))

    # Save -> img_path, [predict scores], predict_score, predict_index, predict_classname
    with open('out.txt', 'w') as f:
        for info in infos:
            f.write('{}\n'.format(info))

    print('..... Finished!......')
