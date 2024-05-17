import torch
from torchvision import models
from ultralytics import YOLO  # 导入 YOLO 模型


def get_model(name="vgg16", pretrained=True, config_file=None):
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
    elif name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
    elif name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
    elif name == "vgg19":
        model = models.vgg19(pretrained=pretrained)
    elif name == "inception_v3":
        model = models.inception_v3(pretrained=pretrained)
    elif name == "googlenet":
        model = models.googlenet(pretrained=pretrained)
    elif name == "YOLOv8":
        if config_file is None:
            raise ValueError("YOLOv8 model requires a config file.")
        model = YOLO(config_file)  # 初始化 YOLOv8 模型
    else:
        raise ValueError(f"Model {name} not recognized. Please choose a valid model name.")

    if torch.cuda.is_available() and name != "YOLOv8":  # YOLO 模型自行处理设备
        return model.cuda()
    else:
        return model
