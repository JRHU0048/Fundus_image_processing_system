import torch

# # 全局中心类，用于管理各个本地站点的模型参数聚合与下发
# class GlobalCenter:
#     def __init__(self, num_clients):
#         self.num_clients = num_clients
#         self.client_models_params = [None] * num_clients
#         self.global_model_params = None

#     def receive_client_params(self, client_id, model_params):
#         """
#         接收本地站点传来的模型参数
#         :param client_id: 本地站点的编号（从0开始）
#         :param model_params: 本地站点训练后的模型参数（模型的state_dict）
#         """
#         self.client_models_params[client_id] = model_params

#     def aggregate_params(self):
#         """
#         对各个本地站点的模型参数进行平均聚合
#         """
#         if None in self.client_models_params:
#             raise ValueError("Not all client parameters have been received.")
#         aggregated_params = {}
#         for key in self.client_models_params[0].keys():
#             values = [self.client_models_params[i][key].float() for i in range(self.num_clients)]
#             aggregated_value = sum(values) / self.num_clients
#             aggregated_params[key] = aggregated_value
#         self.global_model_params = aggregated_params

#     def distribute_params(self):
#         """
#         将聚合后的全局模型参数下发给指定的本地站点
#         :param client_id: 本地站点的编号（从0开始）
#         :return: 下发给该本地站点的模型参数（模型的state_dict）
#         """
#         if self.global_model_params is None:
#             raise ValueError("Global model parameters have not been aggregated yet.")
#         return self.global_model_params
    
#     def init_params(self):
#         """
#         初始化所有客户端的模型参数为0
#         """
#         self.client_models_params = [None] * self.num_clients
      

class GlobalCenter:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.client_params = {}  # 存储每个站点的模型参数
        self.aggregated_params = None  # 存储聚合后的参数

    def receive_client_params(self, client_id, params,data_size):
        """接收来自站点的模型参数"""
        self.client_params[client_id] = {
            'params': params,
            'data_size': data_size
        }
    def aggregate_params(self):
        """聚合所有站点的模型参数（联邦平均）"""
        if not self.client_params:
            raise ValueError("No client parameters received.")

        # 计算总数据量
        total_data = sum(client['data_size'] for client in self.client_params.values())
        
        # 初始化聚合参数
        self.aggregated_params = {}
        for key in self.client_params[0]['params'].keys():
            self.aggregated_params[key] = torch.zeros_like(
                self.client_params[0]['params'][key],
                device=self.client_params[0]['params'][key].device
            )

        # 加权聚合  fedavg
        for client in self.client_params.values():
            client_params = client['params']
            weight = client['data_size'] / total_data
            for key in client_params:
                self.aggregated_params[key] += client_params[key] * weight

    def distribute_params(self):
        """分发聚合后的参数"""
        if self.aggregated_params is None:
            raise ValueError("No aggregated parameters available.")
        return self.aggregated_params

    def init_params(self):
        """初始化参数存储"""
        self.client_params = {}
        self.aggregated_params = None