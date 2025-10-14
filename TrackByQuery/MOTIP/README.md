# MOTIP for MMOT

This repository provides the **MMOT-adapted implementation** of  **[MOTIP](https://arxiv.org/abs/2403.16848).

---

## 🚀 Usage

MOTIP training is divided into **two stages**:
1. **Stage 1** — Detection
2. **Stage 2** — Detection and Association

All configuration files are located in the `configs/` directory.

---

### ⚙️ Step 1: Modify Configuration Parameters

Before training, please edit the following fields in the YAML files.

| File | Parameter | Description |
|------|------------|-------------|
| `configs/base.yaml` | `DATA_ROOT` | Set the absolute path of your MMOT dataset root (e.g. `/data/MMOT/data`). |
| `configs/train_stage1.yaml` | `OUTPUTS_DIR` | Directory for saving checkpoints and logs for Stage 1 training. |
| `configs/train_stage2.yaml` | `OUTPUTS_DIR` | Directory for saving Stage 2 fine-tuning results. |
| `configs/train_stage2.yaml` | `PRETRAIN` | Path to the pretrained weights from Stage  |

### 🧠 Step 2: Stage 1 Training

Train MOTIP with detection-guided supervision.
This stage initializes the Transformer backbone and spectral embedding modules.

```bash

CUDA_VISIBLE_DEVICES=0 python main.py --config-path ./configs/train_3ch_stage1.yaml

#Distribute
CUDA_VISIBLE_DEVICES=0,1 python  -m torch.distributed.run --nproc_per_node=2 main.py --use-distributed True --config-path ./configs/train_3ch_stage1.yaml

#8ch
CUDA_VISIBLE_DEVICES=0 python main.py --config-path ./configs/train_8ch_stage1.yaml

#Distribute
CUDA_VISIBLE_DEVICES=0,1 python  -m torch.distributed.run --nproc_per_node=2 main.py --use-distributed True --config-path ./configs/train_8ch_stage1.yaml

```
After Stage 1, a checkpoint file will be saved in OUTPUTS_DIR (e.g. checkpoint_stage1.pth).

### 🔁 Step 3: Stage 2 Training 

```bash

CUDA_VISIBLE_DEVICES=0 python main.py --config-path ./configs/train_3ch_stage2.yaml

#Distribute
CUDA_VISIBLE_DEVICES=0,1 python  -m torch.distributed.run --nproc_per_node=2 main.py --use-distributed True --config-path ./configs/train_3ch_stage2.yaml

#8ch
CUDA_VISIBLE_DEVICES=0 python main.py --config-path ./configs/train_8ch_stage2.yaml

#Distribute
CUDA_VISIBLE_DEVICES=0,1 python  -m torch.distributed.run --nproc_per_node=2 main.py --use-distributed True --config-path ./configs/train_8ch_stage2.yaml
```

Ensure that the PRETRAIN path in train_stage2.yaml correctly points to
the checkpoint generated in Stage 1.

### Stpe 4: Testing and Evaluating is automatically done during training.

The result is in the `eval_during_train` folder.

## Citation

If you think this project is helpful, please feel free to leave a :star: and cite our paper:

```tex
@article{MOTIP,
  title={Multiple Object Tracking as ID Prediction},
  author={Gao, Ruopeng and Zhang, Yijun and Wang, Limin},
  journal={arXiv preprint arXiv:2403.16848},
  year={2024}
}
```