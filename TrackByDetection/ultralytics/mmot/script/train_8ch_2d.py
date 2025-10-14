import os
from ultralytics import settings
from ultralytics import YOLO

PWD = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = os.path.abspath(os.path.join(PWD, '../../../..'))
# 加载yolo11l模型
pt_file = os.path.join(ROOT_PATH, 'weights/yolo11lint_8ch.pt')
train_cfg = os.path.join(PWD, '../cfg/8ch.yaml')
data_cfg = os.path.join(PWD, '../cfg_data/mmot_8ch.yaml')
model_cfg = os.path.join(PWD, '../../ultralytics/cfg/models/11/yolo11l-obb-8ch.yaml')

experiment_name = 'train_8ch2d' 


model = YOLO(model_cfg).load(pt_file)

results = model.train(
    data=data_cfg,
    epochs=50, 
    device = [4],
    project = os.path.join(ROOT_PATH, 'exp', 'yolo'),
    task = "obb",
    name = experiment_name,
    batch = 4,
    imgsz = 1280,
    cfg = train_cfg,
    workers = 2,
    )