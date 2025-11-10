import os
import random
import numpy as np
from PIL import Image


print(" generate_16bit_image is imported.")
def generate_16bit_image(size=(256, 256), num_targets=5, max_gray_value=17000):
    # 创建一个全零的16位图像
    image = np.zeros(size, dtype=np.uint16)

    # 随机生成目标点的位置
    for _ in range(num_targets):
        # 随机选择中心点位置
        center_x = random.randint(0, size[0] - 1)
        center_y = random.randint(0, size[1] - 1)

        # 随机生成目标的强度
        intensity = random.randint(1000, max_gray_value)  # 避免灰度值为0

        # 随机生成目标的大小
        target_size = random.randint(5, 20)  # 目标大小

        # 生成聚集区域
        for x in range(max(0, center_x - target_size), min(size[0], center_x + target_size)):
            for y in range(max(0, center_y - target_size), min(size[1], center_y + target_size)):
                # 计算到中心的距离
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

                # 根据距离设置灰度值，距离越近，值越高
                if distance < target_size:
                    # 使用高斯分布衰减
                    gray_value = int(intensity * (1 - (distance / target_size)))
                    image[x, y] = min(gray_value, max_gray_value)  # 确保不超过最大值

    return image

def generate_images_in_folder(folder_path, num_images, image_size=(256, 256), num_targets=5, max_gray_value=17000):
    # 创建文件夹，如果不存在的话
    os.makedirs(folder_path, exist_ok=True)

    for i in range(num_images):
        # 生成16位图像
        image_data = generate_16bit_image(size=image_size, num_targets=num_targets, max_gray_value=max_gray_value)

        # 转换为PIL图像并保存
        image_16bit = Image.fromarray(image_data)
        image_path = os.path.join(folder_path, f'image_{i+1:03d}.png')  # 图像命名为 image_001.png, image_002.png, ...
        image_16bit.save(image_path, format='PNG')

        print(f"Generated image {i+1}/{num_images} saved at {image_path}")

# 使用示例
folder_path = 'example/video1'  # 设置目标文件夹路径
num_images = 1000  # 设置生成的图像数量
generate_images_in_folder(folder_path, num_images)
