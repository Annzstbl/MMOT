# MeMOTR for MMOT

This repository provides the **MMOT-adapted implementation** of [MeMOTR](https://arxiv.org/abs/2307.15700) for multispectral multi-object tracking.

---

## 🚀 Usage

### ⚙️ Configuration Files

All configuration files are located in the `configs_mmot/` directory.  
Each file **inherits** from `default.yaml` and overrides specific parameters 

| File | Description |
|------|--------------|
| `default.yaml` | Base configuration file (shared training settings). |
| `train_3ch.yaml` | Training configuration for RGB input (3 channels). |
| `train_8ch_2d.yaml` | Training configuration for 8-channel multispectral input with 2D stem. |
| `train_8ch_3d.yaml` | Training configuration for 8-channel multispectral input with 3D stem. |
| `predict_3ch.yaml` | Testing configuration for 3-channel model. |
| `predict_8ch_2d.yaml` | Testing configuration for 8-channel 2D model. |
| `predict_8ch_3d.yaml` | Testing configuration for 8-channel 3D-stem model. |

## Training

Training is launched via distributed command line execution. You need to change the `DATA_ROOT` in default.yaml, the `OUTPUTS_DIR` and `PRETRAINED_MODEL` in training yaml.
Example commands:

```bash
cd $MMOT_ROOT/TrackByQuery/MeMOTR

python -m torch.distributed.run \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:29502 \
    main.py \
    --use-distributed \
    --config-path /data/users/litianhao/hsmot_code/MeMOTR/configs_mmot/train_8ch_3d.yaml
```

##  Inference

After training, run inference using the corresponding predict configuration. You need to change the `OUTPUTS_DIR` and `SUBMIT_DIR` in predicting yaml.

```bash
python main.py --config-path ./configs_mmot/predict_8ch_3d.yaml

```
The prediction results for each video will be stored under:
```bash
$EXP_DIR/test/tracker
```


You can then evaluate these results using the [TrackEval toolkit](../../TrackEval/Readme.md), which supports HOTA, MOTA, IDF1, and AssA metrics for all MMOT benchmark models.

## Citation
```bibtex
@InProceedings{MeMOTR,
    author    = {Gao, Ruopeng and Wang, Limin},
    title     = {{MeMOTR}: Long-Term Memory-Augmented Transformer for Multi-Object Tracking},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {9901-9910}
}
```