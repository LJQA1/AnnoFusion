# -*- coding: utf-8 -*-
# @Author  : LG

import os
from PIL import Image
import numpy as np
from json import load, dump
from typing import List
import cv2
import base64
from PyQt5 import QtWidgets, QtCore, QtGui
class Object:
    def __init__(self, category:str, group:int, segmentation, area, layer, bbox, iscrowd='', note=''):
        self.category = category
        self.group = group
        self.segmentation = segmentation
        self.area = area
        self.layer = layer
        self.bbox = bbox  # 新增 bbox 属性
        #self.color = color
        self.iscrowd = iscrowd
        self.note = note


def img_to_base64(img_array):
    # 传入图片为RGB格式numpy矩阵，传出的base64也是通过RGB的编码
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) #RGB2BGR，用于cv2编码
    encode_image = cv2.imencode(".jpg", img_array)[1] #用cv2压缩/编码，转为一维数组
    byte_data = encode_image.tobytes() #转换为二进制
    base64_str = base64.b64encode(byte_data).decode("ascii") #转换为base64
    return base64_str


class Annotation:
    def __init__(self, image_path, label_path,dict_list,):
        img_folder, img_name = os.path.split(image_path)
        self.img_folder = img_folder
        self.img_name = img_name
        self.label_path = label_path
        self.note = ''

        self.version = '5.3.1'
        self.flags = ''
        self.imagePath = ''
        self.class_name = [item["name"] for item in dict_list]


        image = np.array(Image.open(image_path))
        if image.ndim == 3:
            self.height, self.width, self.depth = image.shape
        elif image.ndim == 2:
            self.height, self.width = image.shape
            self.depth = 0
        else:
            self.height, self.width, self.depth = image.shape[:, :3]
            print('Warning: Except image has 2 or 3 ndim, but get {}.'.format(image.ndim))
        del image

        self.objects:List[Object,...] = []

    def load_annotation(self,mode):
        if os.path.exists(self.label_path):
            with open(self.label_path, 'r') as f:
                dataset = load(f)
                #info = dataset.get('info', {})
                version = dataset.get('version',None)
                #description = info.get('description', '')
                # if version == '5.1.1':
                if version is not None:
                    objects = dataset.get('shapes', [])

                    width = dataset.get('width', None)
                    if width is not None:
                        self.width = width

                    height = dataset.get('height', None)
                    if height is not None:
                        self.height = height

                    imagePath = dataset.get('imagePath', None)
                    if imagePath is not None:
                        self.imagePath = imagePath
                        #self.imageHeight = imagePath


                    #self.note = '123'
                    for obj in objects:
                        shape_type = obj.get('shape_type', '')

                        if (mode == "Key Point Task" and "polygon" == obj.get('shape_type', 0)):
                            print("Key Point Task can not load polygon")
                            return False
                        elif (mode == "Semantic Segmentation Task" and "point" == obj.get('shape_type',0)):
                            print("Semantic Segmentation Task  can not load polygon")

                        # category = obj.get('label', 'unknow')
                        # group = obj.get('group_id', 0)
                        # if group is None: group = 0
                        # segmentation = obj.get('points', [])
                        # iscrowd = obj.get('shape_type', "polygon")
                        # note = ''
                        # area = ''
                        # layer = 1.0
                        # bbox = ''
                        # obj = Object(category, group, segmentation, area, layer, bbox, iscrowd, note)
                        # self.objects.append(obj)

                        category = obj.get('label', 'unknow')
                        group = obj.get('group_id', 0)
                        if group is None:
                            group = 0
                        points = obj.get('points', [])
                        iscrowd = shape_type
                        note = ''
                        area = ''
                        layer = 1.0

                        # 如果是矩形框标注，将 points 转换为 bbox 格式
                        if shape_type == "rectangle":
                            if len(points) == 2:  # 确保有两个对角点
                                x_min, y_min = points[0]
                                x_max, y_max = points[1]
                                bbox = [x_min, y_min, x_max, y_max]  # 转换为 bbox 格式
                            else:
                                bbox = []  # 如果 points 格式不正确，设置为空
                        else:
                            bbox = []  # 非矩形框标注，bbox 为空

                        # 创建 Object 对象
                        obj = Object(category, group, points, area, layer, bbox, iscrowd, note)
                        self.objects.append(obj)


                else:
                    ####设置一个弹窗终止其他格式的加载####
                    ##在mainwindow设置一个终止弹窗，不加载图像，返回mainwindow的时候判断进行终止####
                    print('Warning: Loaded JSON in another forma.')


    def save_annotation(self, mode, image_data):
        if (len(self.objects) == 0):
            return

        if mode == "Object Detection Task":
            image_data = img_to_base64(image_data)
            dataset = {
                'version': "5.3.1",
                'flags': {},
                'shapes': [],
                'imageData': image_data,
                'imagePath': self.img_name,
                'imageHeight': self.height,
                'imageWidth': self.width
            }

            for obj in self.objects:
                shape = {
                    'label': obj.category,
                    'group_id': obj.group,
                    'description': "",
                    'points': [],
                    'shape_type': "rectangle",
                    'flags': {}
                }

                # 将 bbox 转换为 points 格式
                if obj.bbox:
                    x_min, y_min, x_max, y_max = obj.bbox
                    shape['points'] = [[x_min, y_min], [x_max, y_max]]  # 矩形框的两个对角点

                dataset['shapes'].append(shape)

            with open(self.label_path, 'w') as f:
                dump(dataset, f, indent=4)
            return True

        if (mode == "Semantic Segmentation Task"):
            image_data = img_to_base64(image_data)
            dataset = {}
            dataset['version'] = "5.3.1"
            dataset['flags'] = {}
            dataset['shapes'] = []
            for obj in self.objects:
                shape = {}
                shape['label'] = obj.category
                shape['group_id'] = None
                shape['description'] = ""
                shape['points'] = obj.segmentation
                shape['shape_type'] = "polygon"
                shape['flags'] = {}
                dataset['shapes'].append(shape)
            dataset['imageData'] = image_data
            dataset['imagePath'] = self.img_name
            dataset['imageHeight'] = self.height
            dataset['imageWidth'] = self.width
            with open(self.label_path, 'w') as f:
                dump(dataset, f, indent=4)
            return True

        if (mode == "Key Point Task"):
            image_data = img_to_base64(image_data)
            dataset = {}
            dataset['version'] = "5.3.1"
            dataset['flags'] = {}
            dataset['shapes'] = []
            for obj in self.objects:
                shape = {}
                shape['label'] = obj.category
                shape['points'] = obj.segmentation
                shape['group_id'] = None
                shape['description'] = ""
                shape['shape_type'] = "point"
                shape['flags'] = {}
                dataset['shapes'].append(shape)
            dataset['imageData'] = image_data
            dataset['imagePath'] = self.img_name
            dataset['imageHeight'] = self.height
            dataset['imageWidth'] = self.width
            with open(self.label_path, 'w') as f:
                dump(dataset, f, indent=4)
            return True





