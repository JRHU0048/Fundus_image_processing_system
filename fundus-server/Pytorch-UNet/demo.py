import torchvision.models as models

# 加载 MobileNetV2 的预训练模型
mobilenet_v2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# 打印 features 模块的结构
for idx, layer in enumerate(mobilenet_v2.features):
    print(f"Layer {idx}: {layer}")