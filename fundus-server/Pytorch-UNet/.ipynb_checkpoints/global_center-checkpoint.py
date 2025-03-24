import torch

# 全局中心类，用于管理各个本地站点的模型参数聚合与下发
class GlobalCenter:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.client_models_params = [None] * num_clients
        self.global_model_params = None

    def receive_client_params(self, client_id, model_params):
        """
        接收本地站点传来的模型参数
        :param client_id: 本地站点的编号（从0开始）
        :param model_params: 本地站点训练后的模型参数（模型的state_dict）
        """
        self.client_models_params[client_id] = model_params

    def aggregate_params(self):
        """
        对各个本地站点的模型参数进行平均聚合
        """
        if None in self.client_models_params:
            raise ValueError("Not all client parameters have been received.")
        aggregated_params = {}
        for key in self.client_models_params[0].keys():
            values = [self.client_models_params[i][key].float() for i in range(self.num_clients)]
            aggregated_value = sum(values) / self.num_clients
            aggregated_params[key] = aggregated_value
        self.global_model_params = aggregated_params

    def distribute_params(self):
        """
        将聚合后的全局模型参数下发给指定的本地站点
        :param client_id: 本地站点的编号（从0开始）
        :return: 下发给该本地站点的模型参数（模型的state_dict）
        """
        if self.global_model_params is None:
            raise ValueError("Global model parameters have not been aggregated yet.")
        return self.global_model_params
    
    def init_params(self):
        """
        初始化所有客户端的模型参数为0
        """
        self.client_models_params = [None] * self.num_clients
      
        