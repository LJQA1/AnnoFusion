# -*- coding: utf-8 -*-
# @Author  : LG

import os
import json
from PIL import Image
import numpy as np
from json import load, dump
from typing import List
from configs import STATUSMode, CLICKMode, DRAWMode, CONTOURMode
import cv2
from annotation_labelme import Object

print(" Threshold_projson is imported.")
class Threshold_projson():
    def __init__(self,scene):
        super(Threshold_projson,self).__init__()
        self.scene = scene

    def threshold_save(self,category,label_root,masks_pro,name,image_path):
        self.imagePath = image_path
        image = np.array(Image.open(self.imagePath))
        if image.ndim == 3:
            self.height, self.width, self.depth = image.shape
        elif image.ndim == 2:
            self.height, self.width = image.shape
            self.depth = 0
        self.category = category
        self.name = name

        self.label = os.path.normpath(os.path.join(label_root, os.path.splitext(self.name)[0] + '.json'))

        dataset = {}
        dataset['version'] = '5.3.1'
        dataset['flags'] = {}
        dataset['shapes'] = []
        dataset['imageData'] = ''
        dataset['imagePath'] = self.name
        dataset['imageWidth'] = self.width
        dataset['imageHeight'] = self.height
        dataset['shapes'] = []
        if masks_pro is not None:

            masks = masks_pro
            masks = masks.astype('uint8')
            h, w = masks.shape[-2:]
            masks = masks.reshape(h, w)

            if self.scene.contour_mode == CONTOURMode.SAVE_ALL:
                dst = cv2.GaussianBlur(masks, (3, 3), 0)
                ret, thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                # 当保留所有轮廓时，检测所有轮廓，并建立二层等级关系
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            else:
                dst = cv2.GaussianBlur(masks, (3, 3), 0)
                ret, thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                # 当只保留外轮廓或单个mask时，只检测外轮廓
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for index, contour in enumerate(contours):
                if len(contour) <= 4:
                    continue
                contour = contour.tolist()
                shape = {}
                shape['label'] = self.category
                shape['group_id'] = 0
                shape['points'] = []
                shape['area'] = 0
                shape['bbox'] = []
                shape['layer'] = 1.0
                shape['shape_type'] = "polygon"
                shape['flags'] = None

                for point in contour:
                    x, y = point[0]
                    shape["points"].append([x,y])

                # 提取所有点的坐标
                points = [point[0] for point in contour]
                # 计算最小和最大值
                min_x = min(point[0] for point in points)  # 左上角的 x
                min_y = min(point[1] for point in points)  # 左上角的 y
                max_x = max(point[0] for point in points)  # 右上角的 x
                max_y = max(point[1] for point in points)  # 右下角的 y

                shape['bbox'] = [min_x , min_y, max_x, max_y]
                dataset["shapes"].append(shape)



        print(dataset)  # 打印数据，查看结构和内容
        if not dataset["shapes"]:
            print("No shapes to write, skipping file creation.")
        else:
            print(f"Attempting to write JSON to: {self.label}")
            if os.path.exists(self.label):
                # 读取现有的JSON文件
                with open(self.label, 'r') as f:
                    data = json.load(f)
                # 检查 'shapes' 是否存在，如果不存在则初始化
                if 'shapes' not in data:
                    data['shapes'] = []

                # 将新的 shape 添加到 'shapes' 列表中
                for i in dataset['shapes']:
                    data['shapes'].append(i)

                # 保存修改后的数据到文件
                with open(self.label, 'w') as f:
                    json.dump(data, f, indent=4)
            else:
                with open(self.label, 'w') as f:
                    dump(dataset, f, indent=4)
                print("Write successful.")
        # if not dataset['shapes']:
        #     pass
        # else:
        # #差一个写入的单个文件路径
        #     with open(self.label, 'w') as f:
        #         dump(dataset, f, indent=4)
        # return True


