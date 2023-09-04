from .config import CfgNode as CN

_C = CN()

_C.VERSION = 1

#-----------------------------------
_C.MODEL = CN()
_C.MODEL.MODEL_NAME = 'efficientnet-b2'

#-----------------------------------
_C.TRAIN = CN()
_C.TRAIN.COLOR_STYLE = 'RGB'
_C.TRAIN.TRAIN_BATCH_SIZE = 16
_C.TRAIN.INPUT_SIZE = (256, 256)
_C.TRAIN.MAX_EPOCH = 100
_C.TRAIN.RESUME_EPOCH = 0
_C.TRAIN.GPUS = 2
_C.TRAIN.WEIGHT_DECAY = 5e-4
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.LR = 1e-3
_C.TRAIN.MEAN = [0.407, 0.438, 0.452]
_C.TRAIN.STD = [0.227, 0.215, 0.225]

#-----------------------------------
_C.TEST = CN()
_C.TEST.COLOR_STYLE = 'RGB'
_C.TEST.TEST_BATCH_SIZE = 8
_C.TEST.INPUT_SIZE = (256,256)
_C.TEST.MEAN = [0.407, 0.438, 0.452]
_C.TEST.STD = [0.227, 0.215, 0.225]

#-----------------------------------
_C.DATASET = CN()
_C.DATASET.TRAIN_DIR = ''
_C.DATASET.TEST_DIR = ''
_C.DATASET.NUM_CLASSES= 3
_C.DATASET.IDX_MAP = None