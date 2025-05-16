import os
from ultralytics import settings
from ultralytics import YOLO
from hsmot.load_multi_channel_pt import load_multi_channel_pt


# 加载yolo11l模型
pt_file = 'your/path/to/yolo11l.pt'
train_cfg = '$ROOT_PATH/ultralytics/hsmot/cfg/rgb.yaml'
data_cfg = '$ROOT_PATH/ultralytics/hsmot/cfg_data/hsmot_8ch.yaml'
model_cfg = '$ROOT_PATH/ultralytics/ultralytics/cfg/models/11/yolo11l-obb-8ch.yaml'

experiment_name = 'train_8ch-2d'


model = YOLO(model_cfg).load(pt_file)

# # Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(
    data=data_cfg,
    epochs=50, 
    device = [0],
    project = 'your/path',
    task = "obb",
    name = experiment_name,
    batch = 4,
    imgsz = 1280,
    cfg = train_cfg,
    workers = 2,
    )