

## 眼科疾病诊断系统


## Web端

Web端基于Vue-admin-temple进行开发，部署教程如下：

1、在完成node环境的配置后，进入fundus-web目录下，打开cmd，执行下面的命令下载模块到本地
```
npm install
```
2、在当前目录(./fundus-web)下，执行下面的命令可直接启动项目(默认81端口)
```
npm run serve
```

## 服务端

服务端接口采用FastApi开发，部署教程如下：

1、安装依赖包
```
pip install -r requirement.txt
```

2、进入fundus-server目录下，执行下面命令运行（注：上传代码中未包含模型，因此无法完成检测，但包含完整代码）
```
python main.py
```

---

## 涉及开源项目：
- [![机器视觉相关的完整项目/系统](https://img.shields.io/badge/-机器视觉相关的完整项目/系统-red)](https://github.com/ceresOPA/Computer-Vision-System)
- [![目标检测算法](https://img.shields.io/badge/-YOLOv5-blue)](https://github.com/ultralytics/yolov5)
- [![图像修复](https://img.shields.io/badge/-Bringing--Old--Photos--Back--to--Life-orange)](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life)
- [![图像上色](https://img.shields.io/badge/-colorization-1E88B0)](https://github.com/richzhang/colorization)
- [![前端模板框架](https://img.shields.io/badge/-vue--admin--template-green)](https://github.com/PanJiaChen/vue-admin-template)
