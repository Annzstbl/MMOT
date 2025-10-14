# Association Trackers for MMOT

This directory contains **four association-based multi-object trackers** — **SORT**, **ByteTrack**, **OC-SORT**, and **BoT-SORT** —  which have been **adapted for the MMOT benchmark** to support **rotated bounding box (OBB)** tracking and association.

---


## 🚀 Usage

### 🔧 Step 1: Prepare YOLO Detections

Before running the association trackers, ensure you have generated YOLO detection results  for the **test set** (see [YOLO README](../../ultralytics/README.md)).

### Step2: Predict and Evaluate
You can batch run the association trackers and evaluate their performance using the provided scripts.

You should set ROOT_PATH first and run the `run.sh` script:

```bash
bash run.sh
```

You will get results in `$EXP/tracker/exp_name/track` and evaluation results in `$EXP/tracker/exp_name/eval`.
The evaluation is automatically performed after tracking using [TrackEval toolkit](../../TrackEval/Readme.md).

## Cite
[SORT](https://arxiv.org/abs/2103.04147)

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}
    }

[ByteTrack](https://arxiv.org/abs/2110.06864)

     @article{zhang2021bytetrack,
      title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
      author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
      journal={arXiv preprint arXiv:2110.06864},
      year={2021}
      }

[OC_SORT](https://arxiv.org/abs/2203.14360)

      @inproceedings{cao2023observation,
      title={Observation-centric sort: Rethinking sort for robust multi-object tracking},
      author={Cao, Jinkun and Pang, Jiangmiao and Weng, Xinshuo and Khirodkar, Rawal and Kitani, Kris},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={9686--9696},
      year={2023}
}

[Bot_SORT](https://arxiv.org/abs/2206.14651)

      @article{aharon2022bot,
      title={BoT-SORT: Robust Associations Multi-Pedestrian Tracking},
      author={Aharon, Nir and Orfaig, Roy and Bobrovsky, Ben-Zion},
      journal={arXiv preprint arXiv:2206.14651},
      year={2022}
}

