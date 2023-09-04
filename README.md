### 目前支持的模型
1. efficientnet-b2
2. resnet-50
3. mobilenet-v2

### 创建数据集
- Step1: 下载数据，处理数据格式
```
dataset
├── custom_data
│   ├── class_1
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│   │   ├── ...
│   ├── class_1
│   │   ├── 10001.jpg
│   │   ├── 10002.jpg
│   │   ├── ...
│   ├── ...
```
- Step2: 划分训练集/验证集/测试集

- Step3: 数据预处理，修改运行`tools/data_preprocess.py`

  - 统计训练集的分布
  - 计算训练集的(mean,std)

### 模型训练

- 修改`config/default_config.yaml`配置文件
- 运行`train.py`或`train.sh`开始训练

### 模型测试

- 运行`tools/image.py`，对单张图片进行推理
- 运行`inference.py`，对文件夹下的所有图片进行推理并保存结果到`out.txt`

### 模型评估
- 运行`get_FLOPsAndParams.py`，得到模型的FLOPs和Params信息
- 运行`loss_curve.py`，得到训练损失的变化图

### ONNX转换
- 运行`torch2onnx.py`，对训练得到的pth文件转换为onnx格式


