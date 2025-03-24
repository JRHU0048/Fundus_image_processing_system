# utils 功能
# 1、保存分类模型
# 2、加载分类模型
import torch

def save_model(model, path):
    """
    保存模型
    参数:
        model (nn.Module): 要保存的模型。
        path (str): 模型保存路径。
    """
    torch.save(model.state_dict(), path)

def load_model(model, path, map_location=None):
    """
    加载模型
    参数:
        model (nn.Module): 要加载的模型。
        path (str): 模型保存路径。
        map_location (str or callable): 指定加载设备。
    """
    if map_location is None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用 weights_only=True 提高安全性
    state_dict = torch.load(path, map_location=map_location, weights_only=True)
    
    # 检查模型架构是否匹配
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"模型架构与权重不匹配: {e}")
        raise
    
    model.eval()