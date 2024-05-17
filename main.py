import argparse
import json
import random
import torch
from server import Server
from client import Client
import datasets
import flask
from Flask import flask

if __name__ == '__main__':
    # 解析命令行参数，获取配置文件路径
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', type=str, default='./utils/conf.json', dest='conf')
    args = parser.parse_args()

    # 从配置文件加载配置
    with open(args.conf, 'r', encoding='utf-8') as f:
        conf = json.load(f)

    # 加载训练和评估数据集
    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

    # 获取YOLOv8模型配置文件路径
    config_file = conf.get("config_file")

    # 初始化服务器对象
    server = Server(conf, eval_datasets, config_file=config_file)

    # 初始化客户端对象列表
    clients = []
    for c in range(conf["no_models"]):
        clients.append(Client(conf, train_datasets, c, config_file=config_file))

    print("\n\n")
    # 全局训练循环
    for e in range(conf["global_epochs"]):
        # 从客户端列表中随机选取k个客户端
        candidates = random.sample(clients, conf["k"])
        # 初始化参数累加器
        weight_accumulator = {name: torch.zeros_like(params) for name, params in
                              server.global_model.model.state_dict().items()}

        # 在选定的客户端上进行本地训练并收集参数差异
        for c in candidates:
            diff = c.local_train(server.global_model)
            for name, params in server.global_model.model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        # 汇总参数并更新全局模型
        server.model_aggregate(weight_accumulator)

        # 评估全局模型性能
        acc, loss = server.model_eval()
        print(f"Epoch {e}, acc: {acc:.6f}, loss: {loss:.6f}\n")
