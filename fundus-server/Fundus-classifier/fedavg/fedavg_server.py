import torch
import copy
from collections import OrderedDict

class FedAvgServer:
    def __init__(self, global_model):
        """
        FedAvg中心服务器
        :param global_model: 初始化全局模型
        """
        self.global_model = global_model
        self.global_params = global_model.state_dict()
        
    def aggregate(self, client_params_list, sample_num_list):
        total_samples = sum(sample_num_list)
        
        # 初始化平均参数
        avg_params = OrderedDict()
        for key in self.global_params.keys():
            # 强制转换为float类型保证计算安全
            param = self.global_params[key].float()
            avg_params[key] = torch.zeros_like(param)
            
        # 加权平均
        for params, num in zip(client_params_list, sample_num_list):
            weight = num / total_samples
            for key in params:
                # 显式类型转换
                weighted_param = params[key].float() * weight
                avg_params[key] += weighted_param.to(avg_params[key].dtype)
                
        # 更新全局参数
        new_params = OrderedDict()
        for key in avg_params:
            new_params[key] = avg_params[key].to(self.global_params[key].dtype)
            
        self.global_params = new_params
        self.global_model.load_state_dict(self.global_params)

    def get_global_params(self):
        return copy.deepcopy(self.global_params)