import torch
import models


class Server(object):
    def __init__(self, conf, eval_datasets, config_file=None):
        """
        初始化服务器对象。

        Args:
            conf (dict): 包含配置信息的字典。
            eval_datasets (torch.utils.data.Dataset): 用于评估的数据集。
            config_file (str, optional): YOLOv8配置文件的路径。默认为None。

        Attributes:
            conf (dict): 包含配置信息的字典。
            eval_datasets (torch.utils.data.Dataset): 用于评估的数据集。
            global_model (torch.nn.Module): 全局模型。
            device (torch.device): 用于运行模型的设备。
            optimizer (torch.optim.Optimizer): 模型优化器。
        """
        self.conf = conf
        self.eval_datasets = eval_datasets
        self.global_model = models.get_model(self.conf["model_name"], config_file=config_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.global_model:
            self.global_model = self.global_model.to(self.device)
        self.optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.conf['lr'],
                                         momentum=self.conf['momentum'])

    def model_aggregate(self, weight_accumulator):
        """
        聚合本地客户端模型的参数更新。

        Args:
            weight_accumulator (dict): 包含参数更新的累积器。

        Returns:
            None
        """
        for name, data in self.global_model.state_dict().items():
            update = weight_accumulator[name] / self.conf["k"]
            data.add_(update.to(self.device))

    def model_eval(self):
        """
        评估全局模型的性能。

        Returns:
            acc (float): 准确率。
            total_loss (float): 平均损失。
        """
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.eval_datasets:
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                output = self.global_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                total_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        acc = correct / total
        total_loss /= total
        return acc, total_loss
