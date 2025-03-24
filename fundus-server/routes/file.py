import os
import shutil
import random
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
import subprocess
import cv2
from fastapi import APIRouter, Depends, File, UploadFile
import json
from yolov5 import detect

file_router = APIRouter()

# 文件上传模块
@file_router.post("/photo", summary="上传图片")
async def upload_image(
        file: UploadFile = File(...)
):
    print(f"上传文件:{file.filename}")

    # 本地存储临时方案
    save_dir = "D:/25AI+/Computer-Vision-System/fundus-server/assets" 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print("无文件夹")
    try:
        suffix = Path(file.filename).suffix

        with NamedTemporaryFile(delete=False, suffix=suffix, dir=save_dir) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_file_name = Path(tmp.name).name
    finally:
        file.file.close()

    return {"imageUrl": f"http://127.0.0.1:81/api/assets/{tmp_file_name}",
            "imageName": f"{tmp_file_name}",
            "appImgUrl":f"http://127.0.0.1:8080/assets/{tmp_file_name}"}

# 视频上传可以省略
@file_router.post("/video", summary="上传视频")
async def upload_video(
        file: UploadFile = File(...)
):
    print(f"上传视频:{file.filename}")

    # 本地存储临时方案，一般生产都是使用第三方云存储OSS(如七牛云, 阿里云)
    save_dir = "D:/25AI+/temp_img"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print("无文件夹")
    try:
        suffix = Path(file.filename).suffix

        with NamedTemporaryFile(delete=False, suffix=suffix, dir=save_dir) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_file_name = Path(tmp.name).name
    finally:
        file.file.close()

    return {"videoUrl": f"http://127.0.0.1:81/api/assets/{tmp_file_name}",
            "videoName": f"{tmp_file_name}",
            "appvideoUrl": f"/assets/{tmp_file_name}"}

# 调用YOLO模型进行图片检测!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# @file_router.get("/checkphoto")
# def check_image(model: str,imageName: str):
#     model_cfg = "/assets/"+model+".cfg"
#     server_dir = 'D:/25AI+/Computer-Vision-System/mask-server/' 
#     print("model",model)
#     print("imageName",imageName)
#     if model=='yolov5s' or model == 'mask-yolov5m' or model == 'smoke-yolov5s':
#         print(f"即将执行命令: python {server_dir}/yolov5/detect.py --weights {server_dir}/yolov5/weights/{model}.pt  --source {server_dir}/assets/{imageName} --output {server_dir}/output")
#         os.system(f"python {server_dir}/yolov5/detect.py --weights {server_dir}/yolov5/weights/{model}.pt  --source {server_dir}/assets/{imageName} --output {server_dir}/output")
#         print("模型检测命令执行完毕")
#     else: 
#         # 调用自定义检测方法
#         Data = detect.myDetect(inputSource=f"C:/Users/y2554/Desktop/mask/server/assets/{imageName}",outputPath="C:/Users/y2554/Desktop/mask/server//output",opt_cfg=f"C:/Users/y2554/Desktop/mask/server/yolov3/cfg/{model}.cfg",currentWeights=f"C:/Users/y2554/Desktop/mask/server/yolov3/models/{model}.pt",opt_names="C:/Users/y2554/Desktop/mask/server/yolov3/mask.names")
#         print(Data)
#     return {"masg":"ok",
#             "imageUrl":f"http://127.0.0.1:81/api/output/{imageName}?random={random.randrange(1, 1000)}",
#             "appImgUrl":f"/output/{imageName}?random={random.randrange(1, 1000)}"}

# 配置眼底图像分割任务
@file_router.get("/checkphoto")
def check_image(model: str,imageName: str):
    model_cfg = "/assets/"+model+".cfg"
    server_dir = 'D:/25AI+/Computer-Vision-System/fundus-server/' 
    print("model",model)
    print("imageName",imageName)

    # 分割模型处理逻辑
    if model=='seg_cup':
        print(f"即将执行命令: python {server_dir}/Pytorch-UNet/predict_cup.py  -i {server_dir}/assets/{imageName} -o {server_dir}/output")
        os.system(f"python {server_dir}/Pytorch-UNet/predict_cup.py -i {server_dir}/assets/{imageName} -o {server_dir}/output/{imageName}")
        print("分割模型检测执行完毕")
    elif model=='seg_disc':
        print(f"即将执行命令: python {server_dir}/Pytorch-UNet/predict_disc.py  -i {server_dir}/assets/{imageName} -o {server_dir}/output")
        os.system(f"python {server_dir}/Pytorch-UNet/predict_disc.py -i {server_dir}/assets/{imageName} -o {server_dir}/output/{imageName}")
    elif model=='seg_cup_disc':
        print(f"即将执行命令: python {server_dir}/Pytorch-UNet/predict_cup_disc.py  -i {server_dir}/assets/{imageName} -o {server_dir}/output")
        os.system(f"python {server_dir}/Pytorch-UNet/predict_cup_disc.py -i {server_dir}/assets/{imageName} -o {server_dir}/output/{imageName}")
        print("分割模型检测执行完毕")
    # 分割模型处理逻辑

    # 分类模型处理逻辑
    elif model == 'fundus_classifier':
        try:
            # 使用subprocess捕获输出
            result = subprocess.run(
                [
                    'python', 
                    f'{server_dir}/Fundus-classifier/predict.py',
                    f'{server_dir}/assets/{imageName}'
                ],
                capture_output=True,
                text=True,
                timeout=30  # 设置超时时间
            )
            print("处理完成")
            
            # 解析JSON输出
            if result.returncode == 0:
                output = json.loads(result.stdout)
                print("分类模型检测执行完毕")
                print(output)
                # 修改后的返回部分
                return {
                    "msg": "ok",
                    "predictions": output["predictions"],  # 字段名改为predictions
                    "top_class": output["top_class"],
                    "expert_analysis": output["expert_analysis"]
                }
                
            else:
                return {"error": f"模型执行失败: {result.stderr}"}
                
        except Exception as e:
            return {"error": f"服务端错误: {str(e)}"}
        
    else: 
        print("no model!!!")
    return {"masg":"ok",
            "imageUrl":f"http://127.0.0.1:81/api/output/{imageName}?random={random.randrange(1, 1000)}",
            "appImgUrl":f"/output/{imageName}?random={random.randrange(1, 1000)}"}

# 调用YOLO模型进行视频检测
@file_router.get("/checkvideo")
def check_video(model: str,videoName: str):
    model_cfg = "/assets/"+model+".cfg"
    server_dir = 'C:/Users/YL/Desktop/mask/mask/server/'
    print("model",model)
    print("videoName",videoName)
    if model == 'yolov5s' or model == 'mask-yolov5m' or model == 'smoke-yolov5s':
        os.system(f"python {server_dir}/yolov5/detect.py --weights {server_dir}/yolov5/weights/{model}.pt  --source {server_dir}/assets/{videoName} --output {server_dir}/output")
    else:
        Data = detect.myDetect(inputSource=f"C:/Users/y2554/Desktop/mask/server/assets/{videoName}",outputPath="C:/Users/y2554/Desktop/mask/server/output",opt_cfg=f"C:/Users/y2554/Desktop/mask/server/yolov3/cfg/{model}.cfg",currentWeights=f"C:/Users/y2554/Desktop/mask/server/yolov3/models/{model}.pt",opt_names="C:/Users/y2554/Desktop/mask/server/yolov3/mask.names")
        print(Data)
    # return {"masg":"ok",
    #         "videoUrl":f"http://127.0.0.1:81/api/output/{videoName}?random={random.randrange(1, 1000)}",
    #         "appVideoUrl":f"/output/{videoName}?random={random.randrange(1, 1000)}"}
    return {"masg": "ok",
            "videoUrl": f"http://127.0.0.1:81/api/output/{videoName}?random={random.randrange(1, 1000)}",
            "appVideoUrl": f"/output/{videoName}?random={random.randrange(1, 1000)}"}

@file_router.get("/checkvideoNo")
def check_video(model: str,videoName: str):

    return {"masg": "ok",
            "videoUrl": f"http://127.0.0.1:81/api/output/{videoName}?random={random.randrange(1, 1000)}",
            "appVideoUrl": "https://static-1259365379.cos.ap-chengdu.myqcloud.com/tmpcyvrw84b.mp4"}
# 开启摄像头实时检测
@file_router.get("/camera")
def check_camera(model: str):
    server_dir = 'C:/Users/YL/Desktop/mask/mask/server/'
    if model == 'mask-yolov5s' or model == 'mask-yolov5m' or model == 'smoke-yolov5s':
        os.system(f"python {server_dir}/yolov5/detect.py --weights {server_dir}/yolov5/weights/{model}.pt  --source 0 ")
    else:
        detect.myDetect(inputSource="0", opt_cfg=f"C:/Users/y2554/Desktop/mask/server/yolov3/cfg/{model}.cfg",currentWeights=f"C:/Users/y2554/Desktop/mask/server/yolov3/models/{model}.pt",opt_names="C:/Users/y2554/Desktop/mask/server/yolov3/mask.names")
    return {"msg":"ok"}

# 关闭摄像头实时检测
@file_router.get("/offcamera")
def check_camera():
    sys.exit()
    print("111111111111111111111")
    return {"msg":"ok"}

@file_router.post("/avatar", summary="上传图片")
async def upload_image(
        file: UploadFile = File(...)
):
    print(f"上传文件:{file.filename}")

    # 本地存储临时方案，一般生产都是使用第三方云存储OSS(如七牛云, 阿里云)
    save_dir = "C:/Users/YL/Desktop/mask/mask/server/assets"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print("无文件夹")
    try:
        suffix = Path(file.filename).suffix

        with NamedTemporaryFile(delete=False, suffix=suffix, dir=save_dir) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_file_name = Path(tmp.name).name
    finally:
        file.file.close()

    return {"imageUrl": f"http://127.0.0.1:81/api/assets/{tmp_file_name}",
            "imageName": f"{tmp_file_name}",
            "appImgUrl":f"/assets/{tmp_file_name}"}

# 新增分类处理函数
def classify_image(image_path: str, model_type: str):
    # 加载模型
    if model_type == "my-classifier":
        model = load_classifier("models/weights/classifier.pth")
    
    # 预处理
    img = preprocess(image_path)  # 需要实现与模型匹配的预处理
    
    # 执行推理
    results = model.predict(img)
    
    # 后处理
    class_name, confidence = post_process(results)
    
    # 生成可视化结果（可选）
    vis_path = visualize_results(image_path, class_name)
    
    return {
        "class": class_name,
        "confidence": float(confidence),
        "vis_url": vis_path
    }