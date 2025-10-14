# YOLO for MMOT

This directory provides the YOLO-based detection framework used in the **MMOT benchmark**.  
It supports both **3-channel RGB** and **8-channel multispectral** inputs, serving as the base detector for SORT, ByteTrack, OC-SORT, and BoT-SORT trackers.

---

## 🚀 Usage

### 🔧 Step 1: Convert MMOT dataset to YOLO format

Before training, convert both the `train` and `test` splits of MMOT into the standard YOLO format.  
This step transforms oriented bounding boxes (OBB) and metadata into YOLO-readable labels.

```bash
python ${MMOT_ROOT}/dataset/mot2yolo_obb.py
```
Yolo format labels will be save under
```bash
${MMOT_ROOT}/data/test/yolo_det_labels
${MMOT_ROOT}/data/train/yolo_det_labels
```

🧩 Step 2: Generate global image lists

Use the provided script to gather all image paths into unified list files (train.txt, val.txt, test.txt):

```bash
python dataset/generate_yolo_txt.py
```


These .txt files will be automatically placed under the 
```bash
ultralytics/mmot/cfg_data/test_8ch.txt
ultralytics/mmot/cfg_data/train_8ch.txt
```

🧠 Step 3: Training and Inference

YOLO training and inference scripts are provided in ultralytics/mmot/script/ for different modalities and architectures.

🏋️ Train
```bash
# Train on 3-channel RGB images
python mmot/script/train_3ch.py

# Train on 8-channel MSI (2D backbone)
python mmot/script/train_8ch_2d.py

# Train on 8-channel MSI (3D-stem backbone)
python mmot/script/train_8ch_3d.py
```
🔍 Inference

After training, run YOLO inference to generate detection results used by downstream trackers or MOTRv2:
```bash
python mmot/script/predict_normal \
    ${NPY_PATH} ${OUT_PATH} ${WEIGHTS_FILE} ${MODE}

where:

${NPH_PATH}: path to the .npy (test or train set)

${OUT_PATH}: output folder for detection results

${WEIGHTS_FILE}: pretrained YOLO checkpoint file

${MODE}: input mode (rgb(3ch) or npy(8ch))
```
 For Example
```bash
python mmot/script/predict_normal.py /data/users/litianhao/MMOT/data/train/npy /data/users/litianhao/MMOT/exp/yolo/train_3ch/predict_train /data/users/litianhao/MMOT/exp/yolo/train_3ch/weights/best.pt rgb
python mmot/script/predict_normal.py /data/users/litianhao/MMOT/data/test/npy /data/users/litianhao/MMOT/exp/yolo/train_3ch/predict_test /data/users/litianhao/MMOT/exp/yolo/train_3ch/weights/best.pt rgb
```