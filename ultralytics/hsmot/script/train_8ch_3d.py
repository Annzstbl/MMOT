# import comet_ml
from ultralytics import settings
from ultralytics import YOLO
from hsmot.load_multi_channel_pt import load_convhsi_pt


# 加载yolo11l模型
pt_file = 'your/path/to/yolo11l.pt'
train_cfg = '$ROOT_PATH/ultralytics/hsmot/cfg/8ch.yaml'
data_cfg = '$ROOT_PATH/ultralytics/hsmot/cfg_data/hsmot_8ch.yaml'
model_cfg = '$ROOT_PATH/ultralytics/ultralytics/cfg/models/11/yolo11l-obb-8ch.yaml'
model_cfg = '$ROOT_PATH/ultralytics/ultralytics/cfg/models/11/yolo11l-obb-8ch-convmsi.yaml'

experiment_name = 'train_8ch-3d' 


model = YOLO(model_cfg).load(load_convhsi_pt(pt_file, pt_file.replace('.pt', 'convhsi.pt')))


results = model.train(
    data=data_cfg,
    epochs=50, 
    device = [2],
    project = 'your/path',
    task = "obb",
    name = experiment_name,
    batch = 4,
    imgsz = 1280,
    cfg = train_cfg,
    workers = 2,
    )