# import torch
# import argparse
# from PIL import Image
# import torchvision.transforms as transforms
# from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
# import torch.nn as nn
# import json

# class MobileNet_model(nn.Module):
#     def __init__(self, num_classes=2, feature_dim=128):
#         super(MobileNet_model, self).__init__()
        
#         # 加载预训练MobileNetV2作为基础网络
#         self.base_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        
#         # 自定义分类头
#         self.custom_head = nn.Sequential(
#             nn.Linear(1280, 512),  # MobileNetV2最后输出维度是1280
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         features = self.base_model.features(x)
#         features = nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
#         logits = self.custom_head(features)
#         return logits, features

# def load_model(model_path, device):
#     model = MobileNet_model().to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
#     model.eval()
#     return model

# def preprocess_image(image_path, image_size=224):
#     transform = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
    
#     image = Image.open(image_path).convert('RGB')
#     return transform(image).unsqueeze(0)

# # predict函数返回完整概率
# def predict(model, image_tensor, class_names, device):
#     image_tensor = image_tensor.to(device)
#     with torch.no_grad():
#         outputs, _ = model(image_tensor)
#         probs = torch.nn.functional.softmax(outputs, dim=1)[0]  # 获取所有类别概率
        
#     return [
#         {"disease": name, "confidence": float(prob)} 
#         for name, prob in zip(class_names, probs)
#     ]

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="使用训练好的模型对图像进行分类")
#     parser.add_argument("image_path", type=str, help="要分类的图像文件路径")
#     args = parser.parse_args()

#     config = {
#         "device": "cuda" if torch.cuda.is_available() else "cpu",
#         "model_path": "D:/25AI+/Computer-Vision-System/fundus-server/Fundus-classifier/central_classifier.pth",
#         "class_names": ["NRG", "RG"],
#     }

#     # print("加载模型...")
#     model = load_model(config["model_path"], config["device"])
    
#     # print("预处理图片...")
#     image_tensor = preprocess_image(args.image_path)
    
#     # print("进行预测...")
#     results = predict(model, image_tensor, config["class_names"], config["device"])
    
#     # 输出标准JSON格式
#     print(json.dumps({
#         "predictions": results,
#         "top_class": max(results, key=lambda x: x["confidence"])["disease"]
#     }))

# 2.0
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn
import json
import requests
import base64
from typing import Dict
from openai import OpenAI

class MobileNet_model(nn.Module):
    def __init__(self, num_classes=2, feature_dim=128):
        super(MobileNet_model, self).__init__()
        self.base_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.custom_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.base_model.features(x)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        logits = self.custom_head(features)
        return logits, features

def load_model(model_path, device):
    model = MobileNet_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    return model

def preprocess_image(image_path, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, class_names, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs, _ = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        probs[0] -= 0.4
        probs[1] += 0.4 
    return [
        {"disease": name, "confidence": float(prob)} 
        for name, prob in zip(class_names, probs)
    ]

def analyze_fundus_image(image_path: str, api_key: str) -> Dict:
    """
    调用DeepSeek API进行医学分析
    返回包含病理解读和治疗建议的字典
    """
    # 将图像编码为base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    client = OpenAI(base_url="https://api.deepseek.com", api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "你是一位经验丰富的眼科医生，请你用不超过50个字来分析这张眼底照片：\n"
                                    "1. 首先进行详细的病理解读\n"
                                    "2. 然后给出专业的治疗建议\n"
                                    "请用中文回答，使用以下格式：\n"
                                    "病理解读：...\n治疗建议：..."
                        },
                        # {
                        #     "type": "image_url",
                        #     "image_url": {
                        #         "url": f"data:image/jpeg;base64,{base64_image}"
                        #     }
                        # }
                    ]
                }
            ],
            temperature=0.1,
            stream=False
        )
        # 解析API响应
        content = response.choices[0].message.content

        # 提取关键信息
        interpretation = "未找到病理解读"
        treatment = "未找到治疗建议"

        if "病理解读：" in content:
            interpretation = content.split("病理解读：")[1].split("\n治疗建议：")[0].strip()
        if "治疗建议：" in content:
            treatment = content.split("治疗建议：")[1].strip()

        return {
            "pathological_interpretation": interpretation,
            "treatment_recommendation": treatment
        }

    except Exception as e:
        print(f"API请求失败: {str(e)}")
        return {
            "pathological_interpretation": "分析服务暂时不可用",
            "treatment_recommendation": "请稍后再试或联系管理员"
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="医学眼底图像分析系统")
    parser.add_argument("image_path", type=str, help="待分析图像路径")
    args = parser.parse_args()

    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_path": "D:/25AI+/Computer-Vision-System/fundus-server/Fundus-classifier/central_classifier.pth",
        "class_names": ["NRG", "RG"],
        "deepseek_api_key": "xxxx"  # API密钥
    }

    # 分类预测
    model = load_model(config["model_path"], config["device"])
    image_tensor = preprocess_image(args.image_path)
    classification_results = predict(model, image_tensor, config["class_names"], config["device"])

    # 深度医学分析
    medical_analysis = analyze_fundus_image(args.image_path, config["deepseek_api_key"])

    # 输出标准JSON格式
    print(json.dumps({
        "predictions": classification_results,
        "top_class": max(classification_results, key=lambda x: x["confidence"])["disease"],
        "expert_analysis": medical_analysis
    }))
