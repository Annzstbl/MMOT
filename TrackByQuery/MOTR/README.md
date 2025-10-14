# MOTR for MMOT

This repository provides the **MMOT-adapted implementation** of  
[MOTR: End-to-End Multiple-Object Tracking with TRansformer](https://arxiv.org/pdf/2105.03247.pdf),  
used in our benchmark **MMOT: The First Challenging Benchmark for Drone-based Multispectral Multi-Object Tracking**.

---

## 🚀 Usage

MOTR supports both 3-channel RGB and 8-channel multispectral modes.
All configurations are provided under configs/.

Example training commands:

```bash
# Train on 3-channel RGB input
sh configs/train_mmot_3ch.sh

# Train on 8-channel MSI input
sh configs/train_mmot_8ch.sh

# Train on 8-channel MSI (3D-stem variant)
sh configs/train_mmot_8ch3D.sh
```
These scripts automatically read the dataset paths from train and test under the project root.
Make sure your dataset has been converted to the standard MMOT structure (see main README).

Example testing commands:
```bash
# Test on 3-channel RGB input
sh configs/predict_mmot_3ch_normal.sh $EXP_DIR $GPU_ID

# Test on 8-channel MSI input
sh configs/predict_mmot_8ch_normal.sh $EXP_DIR $GPU_ID

# Test on 8-channel MSI (3D-stem variant)
sh configs/predict_mmot_8ch3D_noramal.sh $EXP_DIR $GPU_ID
```
After testing, inference results for each video will be saved under:
```bash
$EXP_DIR/preds/
```
You can then evaluate these results using the [TrackEval toolkit](../../TrackEval/Readme.md), which supports HOTA, MOTA, IDF1, and AssA metrics for all MMOT benchmark models.



## Citing MOTR
If you find MOTR useful in your research, please consider citing:
```bibtex
@inproceedings{zeng2021motr,
  title={MOTR: End-to-End Multiple-Object Tracking with TRansformer},
  author={Zeng, Fangao and Dong, Bin and Zhang, Yuang and Wang, Tiancai and Zhang, Xiangyu and Wei, Yichen},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```