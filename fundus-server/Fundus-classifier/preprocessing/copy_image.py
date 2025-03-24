import os
import shutil
import pandas as pd

def copy_images_by_labels(label_file_path, image_folder_path, output_folder_path):
    # 读取标签文件
    df = pd.read_excel(label_file_path)
    # 获取标签文件的第8列到最后一列的列名，作为目标文件夹的名称
    class_names = df.columns[7:]

    # 创建目标文件夹
    for class_name in class_names:
        class_folder = os.path.join(output_folder_path, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

    # 遍历标签文件的每一行
    for index, row in df.iterrows():
        # 假设图片文件名存储在第三、四列
        for i in [3, 4]:
            image_name = str(row.iloc[i])
            image_path = os.path.join(image_folder_path, image_name)

            # 检查图片文件是否存在
            if os.path.exists(image_path):
                # 遍历每一个类别标签（从第七列到最后一列）
                for class_name in class_names:
                    if row[class_name] == 1:
                        # 找到对应的类别文件夹
                        class_folder = os.path.join(output_folder_path, class_name)
                        # 复制图片文件到对应的类别文件夹
                        shutil.copy(image_path, os.path.join(class_folder, image_name))
            else:
                print(f"Image {image_name} not found in {image_folder_path}")

if __name__ == "__main__":
    # 标签文件的路径
    label_file_path = "/home/tangzhiri/yanhanhu/framework/data/On-site Test Set/Annotation/on-site test annotation (English).xlsx"
    # 图片文件夹的路径
    image_folder_path = "/home/tangzhiri/yanhanhu/framework/data/OIA-ODIR/preprocessed/on_site_test_set"
    # 输出文件夹的路径
    output_folder_path = "/home/tangzhiri/yanhanhu/framework/data/OIA-ODIR/preprocessed_v2/on_site_test_set"

    copy_images_by_labels(label_file_path, image_folder_path, output_folder_path)