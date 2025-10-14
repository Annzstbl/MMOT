# MOTRv2 for MMOT

This repository provides the **MMOT-adapted implementation** of [MOTRv2: Bootstrapped Multiple Object Tracking with Temporal Transformers](https://arxiv.org/abs/2211.09791), 
used in our benchmark **MMOT: The First Challenging Benchmark for Drone-based Multispectral Multi-Object Tracking**.

MOTRv2 extends MOTR by introducing a **bootstrapped temporal transformer** that enhances inter-frame feature aggregation and improves tracking consistency.  
In this adaptation, MOTRv2 has been reconfigured to support **multispectral UAV data** (3-channel RGB and 8-channel MSI) within the unified MMOT benchmark.

---

## 🚀 Usage

MOTRv2 supports both 3-channel RGB and 8-channel multispectral inputs.  
All configurations are provided under the `configs/` directory.


### Step 1: Prepare YOLO Detections

Before training MOTRv2, you need to train a **YOLO detector** following the instructions in the [`ultralytics/README.md`](../../TrackByDetection/ultralytics/README.md).  
Once the detector is trained:

Run YOLO on the **testing set** to produce predictions.

Then, convert the YOLO detections of testing set and annotations of training set into MOTRv2-compatible JSON files using the provided script:

```bash
# Convert YOLO detections to MOTRv2 JSON format
python yolo_predict2motrv2.py ${YOLO_ANNOTATION_OF_TRAINSET} ${TRAIN_JSON}
python yolo_predict2motrv2.py ${YOLO_PREDICT_RESULT_TESTSET} ${TEST_JSON}

# for example 
python yolo_predict2motrv2.py /data/users/litianhao/MMOT/exp/yolo/train_3ch/predict_train ./train.json
python yolo_predict2motrv2.py /data/users/litianhao/MMOT/exp/yolo/train_3ch/predict_test ./test.json
```

After conversion, update the corresponding DET_DB path in your MOTRv2 configuration YAML files under configs/,
so that MOTRv2 can correctly load the pre-generated detection database.


### Step 2: Example training commands:

```bash
# Train on 3-channel RGB input
sh configs/train_mmot_3ch_20epoch.sh ${GPU}

# Train on 8-channel MSI input(3D-stem)
sh configs/train_mmot_8ch_20epoch.sh ${GPU}

```
These scripts automatically read the dataset paths from train and test under the project root.
Make sure your dataset has been converted to the standard MMOT structure (see main README).

### Step 3: Example testing commands:
```bash
# Test on 3-channel RGB input
sh configs/eval_mmot_3ch.sh.sh $EXP_DIR $GPU_ID

# Test on 8-channel MSI (3D-stem)
sh configs/eval_mmot_8ch.sh $EXP_DIR $GPU_ID
```
After testing, inference results for each video will be saved under:
```bash
$EXP_DIR/submit/
```
You can then evaluate these results using the [TrackEval toolkit](../../TrackEval/Readme.md), which supports HOTA, MOTA, IDF1, and AssA metrics for all MMOT benchmark models.

