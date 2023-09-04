# -*- coding:utf-8 -*-
import argparse
import os
import time
import logging
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from config.config import get_cfg
from data import CustomImageFolderDataset
from data import Compose, ColorStyle, Resize, ToTensor, Normalize
from data import RandomRotate, RandomHorizontalFlip, RandomVerticalFlip, RandomGaussianBlur
from utils import adjust_learning_rate_cosine, adjust_learning_rate_step, LabelSmoothingCrossEntropy
from models import Efficientnet, Resnet, Mobilenet

MODEL_NAMES = {
    'resnet-50': Resnet,
    'resnet-101': Resnet,
    'efficientnet-b2': Efficientnet,
    'mobilenet-v2': Mobilenet,
}


def set_cfg(yaml_file):
    cfg = get_cfg()
    cfg.merge_from_file(yaml_file)
    return cfg

##打印 DATASETS 中的类别数量信息
def dataset_class_info(datasets:torchvision.datasets.ImageFolder) -> None:
    print('Class_name: ', datasets.classes)
    out_dict = {}
    for i in datasets.samples:
        if str(i[1]) not in out_dict:
            out_dict[str(i[1])] = 1
        else:
            out_dict[str(i[1])] += 1
    print('{Label: count_nums}: ', out_dict)

##加载权重文件
def load_checkpoint(filepath): # filepath: 创建训练模型参数保存的文件夹
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    return model

#-------------------------- Val Process --------------------------#
def test(model):
    print('** TEST **')
    model.eval()
    total_correct = 0
    val_iter = iter(test_dataloader)
    max_iter = len(test_dataloader)
    for iteration in range(max_iter):
        try:
            images, labels = next(val_iter)
        except:
            continue
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
            out = model(images)
            prediction = torch.max(out, 1)[1]
            correct = (prediction == labels).sum()
            total_correct += correct
            print('Test iteration: {}/{}'.format(iteration, max_iter),
                  'Accuracy: %.3f' % (correct.float()/cfg.TEST.TEST_BATCH_SIZE))
    print('Test Accuracy: %.3f' %
          (float(total_correct)/(len(test_dataloader) * cfg.TEST.TEST_BATCH_SIZE)))
    return float(total_correct)/(len(test_dataloader) * cfg.TEST.TEST_BATCH_SIZE)


#-------------------------- Train Process --------------------------#
def train(opt, cfg):
    print('# -------- Start Loading --------')
    model_name = cfg.MODEL.MODEL_NAME
    save_folder = opt.saved_path
    os.makedirs(save_folder, exist_ok=True)
    logging.basicConfig(filename=os.path.join(save_folder, 'log.log'), format='%(asctime)-20s%(message)s', filemode='w',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    if not cfg.TRAIN.RESUME_EPOCH:
        info = '****** Training >>{}<< ****** '.format(model_name)
        print(info)
        logging.info(info)
        print('****** loading the Imagenet pretrained weights ****** ')
        model = MODEL_NAMES[model_name](model_name, num_classes=cfg.DATASET.NUM_CLASSES)

    if cfg.TRAIN.RESUME_EPOCH:
        print(' ******* Resume training from >>{}<< epoch {} *********'.format(model_name, cfg.TRAIN.RESUME_EPOCH))
        model = load_checkpoint(os.path.join(save_folder, 'epoch_{}.pth'.format(cfg.TRAIN.RESUME_EPOCH)))

    ## GPU设定
    if cfg.TRAIN.GPUS > 1:
        print('****** using multiple gpus to training ********')
        model = nn.DataParallel(model, device_ids=list(range(cfg.TRAIN.GPUS)))
    else:
        print('****** using single gpu to training ********')
    print("...... Initialize the network done!!! .......")

    if torch.cuda.is_available():
        model.cuda()

    ## 优化器
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    #optimizer = optim.SGD(model.parameters(), lr=cfg.LR,momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

    ## 损失函数
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    criterion = LabelSmoothingCrossEntropy()
    # criterion = nn.MultiLabelSoftMarginLoss
    # criterion = MultiCEFocalLoss(cfg.class_num)

    lr = cfg.TRAIN.LR
    batch_size = cfg.TRAIN.TRAIN_BATCH_SIZE

    # epoch & iter
    epoch_size = len(train_datasets) // batch_size
    max_iter = cfg.TRAIN.MAX_EPOCH * epoch_size
    start_iter = cfg.TRAIN.RESUME_EPOCH * epoch_size
    epoch = cfg.TRAIN.RESUME_EPOCH

    # cosine学习率调整
    warmup_epoch = 5
    warmup_steps = warmup_epoch * epoch_size
    global_step = 0

    # step学习率调整参数
    stepvalues = (max_iter*0.6, max_iter*0.8)
    step_index = 0

    print('# -------- Start Training --------')
    max_test_acc = 0  # 用来判断best.pth
    model.train()
    last_time = time.time()
    for iteration in range(start_iter, max_iter):
        global_step += 1
        ## 保存 pth 文件
        if iteration % epoch_size == 0:   # 每一个Epoch训练开始时
            # create batch iterator
            batch_iterator = iter(train_dataloader)
            loss = 0
            epoch += 1
            # ------------> 保存best.pth
            if epoch > 1:
                test_acc = test(model)
                # 保存best.pth权重
                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                    # checkpoint = {'model': model.module,
                    #               'model_state_dict': model.module.state_dict(),
                    #               # 'optimizer_state_dict': optimizer.state_dict(),
                    #               'epoch': epoch}
                    checkpoint = {'model': model,
                                  'model_state_dict': model.state_dict(),
                                  # 'optimizer_state_dict': optimizer.state_dict(),
                                  'epoch': epoch}
                    torch.save(checkpoint, os.path.join(
                        save_folder, 'best.pth'.format(epoch)))

                model.train()
            # ------------> 每5轮保存 epoch_5*.pth
            if epoch % 5 == 0 and epoch > 0:
                if cfg.TRAIN.GPUS > 1:
                    checkpoint = {'model': model.module,
                                  'model_state_dict': model.module.state_dict(),
                                  # 'optimizer_state_dict': optimizer.state_dict(),
                                  'epoch': epoch}
                    torch.save(checkpoint, os.path.join(
                        save_folder, 'epoch_{}.pth'.format(epoch)))
                else:
                    checkpoint = {'model': model,
                                  'model_state_dict': model.state_dict(),
                                  # 'optimizer_state_dict': optimizer.state_dict(),
                                  'epoch': epoch}
                    torch.save(checkpoint, os.path.join(
                        save_folder, 'epoch_{}.pth'.format(epoch)))

        ## 调整学习率
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate_step(
            optimizer, cfg.TRAIN.LR, 0.1, epoch, step_index, iteration, epoch_size)

        ## 调整学习率
        # lr = adjust_learning_rate_cosine(optimizer, global_step=global_step,
        #                           learning_rate_base=cfg.TRAIN.LR,
        #                           total_steps=max_iter,
        #                           warmup_steps=warmup_steps)

        ## Forward and Backward
        images, labels = next(batch_iterator)
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        out = model(images)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## Get and Print result
        prediction = torch.max(out, 1)[1]
        train_correct = (prediction == labels).sum()
        train_acc = (train_correct.float()) / batch_size
        if iteration % 10 == 0: # (every 10 iters)
            end_time = time.time()
            used_time = end_time - last_time
            last_time = end_time
            info_t = 'Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size) + \
                        '|| Totel iter ' + repr(iteration) + ' || Loss: %.5f||' % (loss.item()) + 'Accuracy: %.3f ||' % (train_acc * 100) + 'LR: %.8f ||' % (lr) + 'TM: %.3fs' % used_time
            print(info_t)
            logging.info(info_t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Classification baseline!!')
    parser.add_argument('--cfg', type=str, default='config/default_config.yaml',help='config path')
    parser.add_argument('--saved_path', type=str,default='weights/test/')
    opt = parser.parse_args()
    cfg = set_cfg(opt.cfg)
    print('# -------- Configs --------')
    print(cfg)

    ## 数据增强部分
    train_transform = Compose([
        ColorStyle(cfg.TRAIN.COLOR_STYLE),  # 'RGB' / 'Gray'
        Resize(cfg.TRAIN.INPUT_SIZE),
        # RandomVerticalFlip(),
        # RandomHorizontalFlip(),
        # RandomRotate(90, 0.3),
        # RandomGaussianBlur(),
        ToTensor(),
        Normalize(mean=cfg.TRAIN.MEAN, std=cfg.TRAIN.STD),
    ])
    test_transform = Compose([
        ColorStyle(cfg.TEST.COLOR_STYLE),
        Resize(cfg.TEST.INPUT_SIZE),
        ToTensor(),
        Normalize(mean=cfg.TRAIN.MEAN, std=cfg.TRAIN.STD),
    ])

    ##ImageFolder对象可以将一个文件夹下的文件构造成一类, 所以数据集的存储格式为一个类的图片放置到一个文件夹下
    ##然后利用dataloader构建提取器，每次返回一个batch的数据，在很多情况下，利用num_worker参数设置多线程，来相对提升数据提取的速度
    print('# -------- Data Infos --------')
    train_datasets = CustomImageFolderDataset(root=cfg.DATASET.TRAIN_DIR, transform=train_transform)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    print('## Train Data Info: ')
    dataset_class_info(train_datasets)
    test_datasets = CustomImageFolderDataset(root=cfg.DATASET.TEST_DIR,transform=test_transform)
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=cfg.TEST.TEST_BATCH_SIZE, shuffle=True, num_workers=0)
    print('## Test Data Info: ')
    dataset_class_info(test_datasets)

    ## 正式训练
    train(opt, cfg)
