import os
import numpy as np
from ultralytics import YOLO
import json
import base64
import cv2


print(" output is imported.")
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img
def img_to_base64(img_array):
    # 传入图片为RGB格式numpy矩阵，传出的base64也是通过RGB的编码
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) #RGB2BGR，用于cv2编码
    encode_image = cv2.imencode(".jpg", img_array)[1] #用cv2压缩/编码，转为一维数组
    byte_data = encode_image.tobytes() #转换为二进制
    base64_str = base64.b64encode(byte_data).decode("ascii") #转换为base64
    return base64_str
def read_json(path):
    with open(path, 'r') as f:
        content = f.read()
    return json.loads(content)

def output_label(image_path,
                 box_cls,
                 keypoint_cls,
                 keypoints,
                 labels,
                 save_dir:str = "key_points",
                 is_save_json:bool = False ):
    '''
      这里默认 图像 位置与 mask json 的位置在同一文件下
      生成的 同级目录文件 key_points 下
      keypoint : [[x1,y1],[x2,y2].....]
      labels : ["nose", "eye",.....]

      问题:
      mask json文件路径是否可选择
      保存标注文件路径是否可选择
      keypoint, labels 格式问题
    '''

    # 判断有无关键点标注
    if (len(labels) <= 0):
        print("未标注任何关键点！")
        return

    # 判断是否存在文件保存路径
    dir_path = ""
    dir_lists = image_path.split("/")[:-1]
    for dir_list in dir_lists:
        dir_path += dir_list + "/"
    """
        save_path
    """
    save_path = os.path.join(dir_path, save_dir)
    if(not os.path.exists(save_path)):
        os.mkdir(save_path)

    # 判断是否存在 mask json文件
    """
        dir_path_json
    """
    dir_path_json = dir_path
    name = image_path.split("/")[-1].split(".")[0]
    mask_json_path = os.path.join(dir_path_json, name + ".json")
    assert os.path.exists(mask_json_path), mask_json_path + "文件不存在"

    # 判断mask json文件中是否存在多边形
    labelme = read_json(mask_json_path)
    num = [this_target for this_target in labelme['shapes'] if(this_target['shape_type'] == 'polygon')]
    if(len(num) <= 0):
        print(mask_json_path + "json文件中没有mask标注,无法按照格式生成txt文件!")
        return

    img_width = labelme['imageWidth']
    img_height = labelme['imageHeight']
    txt_name = name + ".txt"
    json_name = name + ".json"
    save_txt_path = os.path.join(save_path, txt_name)
    save_json_path = os.path.join(save_path, json_name)
    with open(save_txt_path, 'w', encoding='utf-8') as f:
        for i,each_ann in enumerate(labelme['shapes']):
            label_box = each_ann['label']
            if each_ann['shape_type'] == 'polygon':
                yolo_str = ''
                box_class_id = box_cls.index(each_ann['label'])
                yolo_str += '{} '.format(box_class_id)
                box_top_left_x = int(min(np.array(each_ann['points'])[:,0]))
                box_bottom_right_x = int(max(np.array(each_ann['points'])[:,0]))
                box_top_left_y = int(min(np.array(each_ann['points'])[:,1]))
                box_bottom_right_y = int(max(np.array(each_ann['points'])[:,1]))

                if(not is_save_json):
                    box_center_x = int((box_top_left_x + box_bottom_right_x) / 2)
                    box_center_y = int((box_top_left_y + box_bottom_right_y) / 2)
                    box_width = box_bottom_right_x - box_top_left_x
                    box_height = box_bottom_right_y - box_top_left_y
                    box_center_x_norm = box_center_x / img_width
                    box_center_y_norm = box_center_y / img_height
                    box_width_norm = box_width / img_width
                    box_height_norm = box_height / img_height
                    yolo_str += '{:.5f} {:.5f} {:.5f} {:.5f} '.format(box_center_x_norm, box_center_y_norm,
                                                                      box_width_norm, box_height_norm)

                    box_keypoints_dict = {}
                    for i, label in enumerate(labels):
                        x, y = keypoints[i]
                        if (x > box_top_left_x) & (x < box_bottom_right_x) & (y < box_bottom_right_y) & (
                                y > box_top_left_y):
                            box_keypoints_dict[label] = [x, y]
                    for each_class in keypoint_cls:
                        if each_class in box_keypoints_dict:
                            keypoint_x_norm = box_keypoints_dict[each_class][0] / img_width
                            keypoint_y_norm = box_keypoints_dict[each_class][1] / img_height
                            yolo_str += '{:.5f} {:.5f} {} '.format(keypoint_x_norm, keypoint_y_norm, 2)
                        else:
                            yolo_str += '0 0 0 '
                    f.write(yolo_str + '\n')
                else:
                    out = {"version": "5.3.1", "flag": {}, "shapes": []}
                    shape = {}
                    shape["label"] = label_box
                    shape["points"] = np.array([[box_top_left_x, box_top_left_y],
                                                [box_bottom_right_x, box_bottom_right_y]]).tolist()
                    shape["group_id"] = 'null'
                    shape["description"] = ""
                    shape['shape_type'] = "rectangle"
                    shape["flag"] = {}
                    out["shapes"].append(shape)
                    if(i == len(labelme['shapes']) - 1):
                        for i, label in enumerate(labels):
                            this_point = keypoints[i]  # [x, y]
                            shape = {}
                            shape["label"] = keypoint_cls[label]
                            shape["points"] = np.array([this_point]).tolist()
                            shape["group_id"] = 'null'
                            shape["description"] = ""
                            shape['shape_type'] = "point"
                            shape["flag"] = {}
                            out["shapes"].append(shape)
                        image = cv_imread(image_path)
                        out["imageData"] = img_to_base64(image)
                        out['imagePath'] = image_path.split('/')[-1]
                        out['imageHeight'] = img_height
                        out['imageWidth'] = img_width
                        with open(save_json_path, 'w') as f:
                            json.dump(out, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    keypoint_cls = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
                    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    boxes_cls = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                 'fire hydrant','stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                 'bear', 'zebra','giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                 'sports ball','kite','baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
                 'cup', 'fork','knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                 'pizza','donut','cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote','keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                 'vase','scissors','teddy bear', 'hair drier', 'toothbrush']
    keypoint = [[100,200],[500,300],[800,500],[600,400],[600,700]]
    labels = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
    image_path = "C:/Users/Admin/Desktop/image/000000000113.jpg"
    output_label(image_path,boxes_cls, keypoint_cls,keypoint, labels)

