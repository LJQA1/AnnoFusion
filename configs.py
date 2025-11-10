import yaml
from enum import Enum


DEFAULT_CONFIG_FILE = 'default.yaml'
CONFIG_FILE = {
    'Semantic Segmentation Task': 'base.yaml',
    'Key Point Task': 'key.yaml',
    'Object Detection Task': 'detection.yaml',  # 新增目标检测任务配置文件--------------1-------------------------------------------------------
    '0':'isat.yaml'
}
CONFIG_FILES = {
    'Semantic Segmentation Task': 'isat.yaml',
    'Key Point Task': 'base.yaml',
    '0':'base.yaml'
}

def load_config(file):
    with open(file, 'rb') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    return cfg

def save_config(cfg, file):
    s = yaml.dump(cfg)
    with open(file, 'w') as f:
        f.write(s)
    return True

class STATUSMode(Enum):
    VIEW = 0
    CREATE = 1
    EDIT = 2

class DRAWMode(Enum):
    POLYGON = 0
    SEGMENTANYTHING = 1
    BOUNDINGBOX = 2  # 新增目标检测绘制模式---------------------------1-------------------------------------------------------------------

class CLICKMode(Enum):
    POSITIVE = 0
    NEGATIVE = 1

class MAPMode(Enum):
    LABEL = 0
    SEMANTIC = 1
    INSTANCE = 2

class CONTOURMode(Enum):
    SAVE_MAX_ONLY = 0       # 只保留最多顶点的mask（一般为最大面积）
    SAVE_EXTERNAL = 1       # 只保留外轮廓
    SAVE_ALL = 2            # 保留所有轮廓