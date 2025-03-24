import cv2
import numpy as np
import os

def auto_crop_fundus_image(image):
    """
    对眼底图像进行自动裁剪的函数

    参数:
    image (numpy.ndarray): 输入的彩色眼底图像

    返回:
    cropped_image (numpy.ndarray): 裁剪后的图像
    """
    # 步骤1: 将彩色图像转换为灰度图像
    # 对于白色像素，像素值设为255；对于黑色像素，像素值设为0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

    # 步骤2: 生成用于裁剪的掩码
    # 如果像素值大于指定的容差，掩码值设为1 (True)；否则设为0 (False)
    tolerance = 6
    mask = np.where(binary_image > tolerance, 1, 0)

    # 步骤3: 识别包含像素值为1的行和列的矩形区域
    rows, cols = np.where(mask == 1)
    top = np.min(rows)
    bottom = np.max(rows)
    left = np.min(cols)
    right = np.max(cols)

    # 步骤4: 从RGB格式的图像中提取识别出的矩形区域
    cropped_image = image[top:bottom + 1, left:right + 1]

    return cropped_image


def resize_image(image, target_size=(224, 224)):
    """
    将图像Resize到指定大小

    参数:
    image (numpy.ndarray): 输入图像
    target_size (tuple): 目标大小，默认为(224, 224)

    返回:
    resized_image (numpy.ndarray): Resize后的图像
    """
    return cv2.resize(image, target_size)


def enhance_contrast(image):
    """
    使用CLAHE方法增强图像对比度

    参数:
    image (numpy.ndarray): 输入的灰度图像

    返回:
    enhanced_image (numpy.ndarray): 对比度增强后的图像
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image


def reduce_noise(image):
    """
    使用中值滤波降噪

    参数:
    image (numpy.ndarray): 输入图像

    返回:
    denoised_image (numpy.ndarray): 降噪后的图像
    """
    return cv2.medianBlur(image, 3)


def preprocess_image(image_path):
    """
    对输入图像进行完整的数据预处理流程

    参数:
    image_path (str): 图像文件路径

    返回:
    preprocessed_image (numpy.ndarray): 预处理后的图像
    """
    # 读取彩色图像
    image = cv2.imread(image_path)

    # 圆形边框裁剪
    cropped_image = auto_crop_fundus_image(image)

    # 图像Resize
    resized_image = resize_image(cropped_image)

    # 转换为灰度图像，用于对比度增强和降噪
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # 图像对比度增强
    enhanced_image = enhance_contrast(gray_image)

    # 降噪
    denoised_image = reduce_noise(enhanced_image)

    # 转换回彩色图像(三通道)
    preprocessed_image = cv2.cvtColor(denoised_image, cv2.COLOR_GRAY2BGR)

    # 检查图像是否为三通道
    # if len(preprocessed_image.shape) == 3 and preprocessed_image.shape[2] == 3:
    #     print("图像是三通道的")
    #     # 检查三个通道的像素值是否相同（判断是否为三通道灰度图）
    #     channel1 = preprocessed_image[:, :, 0]
    #     channel2 = preprocessed_image[:, :, 1]
    #     channel3 = preprocessed_image[:, :, 2]
    #     if np.array_equal(channel1, channel2) and np.array_equal(channel2, channel3):
    #         print("图像是三通道的灰度图")
    #     else:
    #         print("图像是三通道且不是灰度图")
    # else:
    #     print("图像不是三通道的")

    return preprocessed_image


# 输入文件夹路径，包含要处理的图片
input_folder = "/home/tangzhiri/yanhanhu/framework/data/Off-site Test Set/Images"
# 输出文件夹路径，用于保存处理后的图片
output_folder = "/home/tangzhiri/yanhanhu/framework/data/OIA-ODIR/preprocessed/off_site_test_set"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # 处理常见的图像格式
        image_path = os.path.join(input_folder, filename)
        preprocessed_image = preprocess_image(image_path)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, preprocessed_image)

print("所有图片处理并保存完成。")