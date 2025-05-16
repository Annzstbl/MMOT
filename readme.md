# Installation and Setup

## Repository Structure

The codebase is organized into the following key directories:

```text
├── ByteTrack  /            # implementation of sort/bytetrack/oc-sort/bot-sort
├── dataset/                # Data preprocessing and conversion scripts
├── hsmot/                  # Core tracking modules and utilities
├── MOTR/                   # Implementation of MOTR models
├── MOTRv2/                 # Implementation of MOTRv2 models
├── MeMOTR/                 # Implementation of MeMOTR models
├── MOTIP/                  # Implementation of MOTIP architecture
├── ultralytics/            # YOLO-based training and prediction framework
├── TrackEval/              # Official evaluation scripts for MOT metrics
├── workdir/                # Symlinked workspace for experiment outputs
├── data/                   # Symlink to dataset directory
└── README.md               # Project documentation
```

Each method (MOTR, MOTRv2, MeMOTR, MOTIP, YOLO) has its own training and evaluation interface, typically under its corresponding folder. Configuration files must be adapted with correct dataset/model paths before execution.

## Dataset Preparation

The full dataset can be downloaded from the following Hugging Face repository:

👉 [MMOT Dataset on Hugging Face](https://huggingface.co/datasets/Annzstbl/MMOT/tree/main)

Please extract the downloaded `.tar` files and organize them into the following structure. You can use the script `dataset/multitxt_to_one.py` to merge the annotation files provided on Hugging Face into individual per-video `.txt` files:

```text
ROOT_PATH
├── data
│   ├── train
│   │   ├── npy
│   │   │   ├── vid-1
│   │   │   │   ├── 000001.npy
│   │   │   │   └── 000002.npy
│   │   │   └── vid-2
│   │   └── mot
│   │       ├── vid-1.txt
│   │       └── vid-2.txt
│   └── test
│       ├── npy
│       └── mot
```

Link and organize your datasets as follows:

```bash
# Link dataset
ln -s /path/to/dataset ./data 

# Link working directory(optional)
ln -s /path/to/workdir ./workdir
```

## Pretrained Models

All pretrained models used in this project can be downloaded from the following Google Drive link:

👉 [Pretrained Models Folder](https://drive.google.com/drive/folders/1IT0CB7Xdyo7Nbbm7d-xqEqlRe2jB3e26)

Please download the required checkpoints and place them into the appropriate subfolders or update your config paths accordingly before running any training or inference script.

## Environment Setup

Due to compatibility across multiple algorithms, separate environments are required:

### MMOT1 Environment (for MOTR, MOTRv2, MeMOTR)

Create the MMOT1 environment:

```bash
conda create -n mmot1 python=3.10
conda activate mmot1

pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install "numpy<2.0"
pip install -U openmim
mim install mmcv==2.2.0
pip install -r requirements.txt

# Install local packages (update ROOT_PATH before execution)
cd $ROOT_PATH/hsmot
pip install -v -e .

# Compile Deformable Attention operations
cd $ROOT_PATH/MOTR/models/ops
sh make.sh
```

### MMOT2 Environment (for MOTIP)

Create the MMOT2 environment:

```bash
conda create -n mmot2 python=3.11
conda activate mmot2

conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install matplotlib pyyaml scipy tqdm tensorboard seaborn scikit-learn pandas
pip install opencv-python einops wandb pycocotools timm
pip install "numpy<2.0"
pip install -U openmim
mim install mmcv==2.2.0
pip install -r requirements.txt

# Install local packages (update ROOT_PATH before execution)
cd $ROOT_PATH/hsmot
pip install -v -e .

# Compile Deformable Attention operations
cd $ROOT_PATH/MOTIP/models/ops/
sh make.sh
```

### YOLO Environment

Create the YOLO environment:

```bash
conda create -n yolo python=3.10
conda activate yolo
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

# Install local packages (update ROOT_PATH before execution)
cd $ROOT_PATH/hsmot
pip install -v -e .

cd $ROOT_PATH/ultralytics
pip install -v -e .
```

## Training and Evaluation

### YOLO + SORT/BYTETRACK/OC-SORT/BOT-SORT

Modify the absolute paths in each Python file below before running. Ensure pretrained weights are placed correctly:

```bash
# Prepare data
python dataset/mot2yolo_obb.py
python dataset/generate_yolo_txt.py

# Train
python ultralytics/hsmot/script/train_3ch.py
python ultralytics/hsmot/script/train_8ch_2d.py
python ultralytics/hsmot/script/train_8ch_3d.py

# Inference
python ultralytics/hsmot/script/predict_normal $NPH_PATH$ $OUT_PATH$ $WEIGHTS_FILE$ $MODE$ (rgb/npy)

# Evaluation
sh Bytetrack/run.sh
```

### MOTR

Modify the absolute paths in each shell script before running. Ensure pretrained weights are placed accordingly:

```bash
# Training
sh MOTR/configs/train_hsmot_3ch.sh
sh MOTR/configs/train_hsmot_8ch.sh
sh MOTR/configs/train_hsmot_8ch_3d.sh

# Inference
sh MOTR/configs/predict_hsmot_3ch_normal.sh $EXP_DIR$ $GPU$
sh MOTR/configs/predict_hsmot_8ch_normal.sh $2D_EXP_DIR$ $GPU$
sh MOTR/configs/predict_hsmot_8ch_normal.sh $3D_EXP_DIR$ $GPU$

# Evaluation
sh TrackEval/eval.sh motr/train_3ch preds $EXP_ROOT$
sh TrackEval/eval.sh motr/train_8ch preds $EXP_ROOT$
sh TrackEval/eval.sh motr/train_8ch_3d preds $EXP_ROOT$
```

### MOTRv2

Modify the absolute paths in each shell script and Python file before running. Download the proper pretrained models:

```bash
# Data preparation
python MOTRv2/yolo_predict2motrv2.py $YOLO_PREDICT_RESULT_TRAINSET$ $TRAIN_JSON$
python MOTRv2/yolo_predict2motrv2.py $YOLO_PREDICT_RESULT_TESTSET$ $TEST_JSON$

# Training
sh MOTRv2/configs/train_hsmot_3ch_20epoch.sh $GPU$
sh MOTRv2/configs/train_hsmot_8ch_20epoch.sh $GPU$

# Inference
sh MOTRv2/configs/eval_hsmot_rgb.sh $EXP_DIR$ $GPU$
sh MOTRv2/configs/eval_hsmot_8ch.sh $EXP_DIR$ $GPU$

# Evaluation
sh TrackEval/eval.sh motrv2/train_3ch preds $EXP_ROOT$
sh TrackEval/eval.sh motrv2/train_8ch preds $EXP_ROOT$
```

### MeMOTR

Update absolute paths in each YAML configuration file before executing. Make sure weights are downloaded from the pretrained model folder:

```bash
# Training
python MeMOTR/main.py --config-path ./MeMOTR/configs_hsmot/train_3ch.yaml
python MeMOTR/main.py --config-path ./MeMOTR/configs_hsmot/train_8ch_2d.yaml
python MeMOTR/main.py --config-path ./MeMOTR/configs_hsmot/train_8ch_3d.yaml

# Inference
python MeMOTR/main.py --config-path ./MeMOTR/configs_hsmot/predict_3ch.yaml
python MeMOTR/main.py --config-path ./MeMOTR/configs_hsmot/predict_8ch_2d.yaml
python MeMOTR/main.py --config-path ./MeMOTR/configs_hsmot/predict_8ch_3d.yaml

# Evaluation
sh TrackEval/eval.sh memotr/train_3ch tracker $EXP_ROOT$
sh TrackEval/eval.sh memotr/train_8ch_2d tracker $EXP_ROOT$
sh TrackEval/eval.sh memotr/train_8ch_3d tracker $EXP_ROOT$
```

### MOTIP

All configuration paths should be updated prior to execution. Pretrained DETR checkpoints should be downloaded beforehand if resuming Stage 2 directly.

```bash
# Stage 1: DETR-only training (no temporal modeling)
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 main.py \
  --use-distributed True --config-path ./MOTIP/configs/train_3ch_stage1.yaml

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 main.py \
  --use-distributed True --config-path ./MOTIP/configs/train_8ch_stage1.yaml

# Stage 2: Full MOTIP training + inference + evaluation
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 main.py \
  --use-distributed True --config-path ./MOTIP/configs/train_3ch_stage2.yaml

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 main.py \
  --use-distributed True --config-path ./MOTIP/configs/train_8ch_stage2.yaml
```



# Acknowledgements

We sincerely thank the contributors and authors of the following open-source projects that served as the backbone or inspiration for our implementations:

* [YOLO](https://github.com/ultralytics/ultralytics)
* [SORT](https://github.com/abewley/sort)
* [ByteTrack](https://github.com/ifzhang/ByteTrack)
* [OC-SORT](https://github.com/noahcao/OC_SORT)
* [BoT-SORT](https://github.com/yezzed/BoT-SORT)
* [TrackEval](https://github.com/JonathonLuiten/TrackEval)
* [MOTR](https://github.com/megvii-research/MOTR)
* [MOTRv2](https://github.com/megvii-research/MOTRv2)
* [MeMOTR](https://github.com/MCG-NJU/MeMOTR)
* [MOTIP](https://github.com/MCG-NJU/MOTIP)

Their open-source efforts have significantly accelerated research progress in the field of multi-object tracking.
