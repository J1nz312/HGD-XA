import torch
from tqdm import tqdm
from ultralytics import YOLO


class Client(object):
    def __init__(self, conf, train_dataset, id=-1, config_file=None):
        """
        初始化客户端对象。

        参数：
            conf (dict): 客户端配置信息。
            train_dataset (torch.utils.data.Dataset): 训练数据集。
            id (int, optional): 客户端ID，默认为-1。
            config_file (str, optional): YOLOv8模型配置文件路径，默认为None。

        异常：
            ValueError: 当config_file为None时抛出异常。
        """
        self.conf = conf
        self.client_id = id
        self.train_dataset = train_dataset

        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        train_indices = all_range[id * data_len: (id + 1) * data_len]

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            train_indices))

        # 初始化 YOLOv8 模型
        if config_file is None:
            raise ValueError("YOLOv8 model requires a config file.")
        self.local_model = YOLO(config_file)

    def local_train(self, server_global_model):
        """
        在本地训练客户端模型。

        参数：
            server_global_model (Server): 全局模型对象。

        返回：
            diff (dict): 包含本地模型参数与全局模型参数差异的字典。
        """
        # 将全局模型参数复制到本地模型
        self.local_model.model.load_state_dict(server_global_model.model.state_dict())

        # 使用SGD优化器
        optimizer = torch.optim.SGD(self.local_model.model.parameters(), lr=self.conf['lr'],
                                    momentum=self.conf['momentum'])

        self.local_model.model.train()
        for e in range(self.conf["local_epochs"]):
            for batch_id, batch in enumerate(tqdm(self.train_loader)):
                data, target = batch

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = self.local_model(data)  # 使用 YOLO 模型的推理方法
                # 在模型推理后，将输出列表合并成一个张量
                outputs = torch.cat(output, dim=0)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

            print(f"Epoch {e} done.")

        # 添加差分隐私保护
        noise_multiplier = self.conf['noise_multiplier']
        max_grad_norm = self.conf['max_grad_norm']

        diff = {}
        for name, data in self.local_model.model.state_dict().items():
            noise = torch.normal(0, noise_multiplier, size=data.size(), device=data.device)
            diff[name] = (data - server_global_model.model.state_dict()[name]) + noise
            diff[name] = torch.clamp(diff[name], -max_grad_norm, max_grad_norm)

        return diff
