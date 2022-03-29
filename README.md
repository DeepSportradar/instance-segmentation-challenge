- [ ] Training
- [ ] Generation of dataset
- [ ] Pickle to COCO annotation format


# DeepSportRadar Instance Segmentation Challenge

## Installation

### Downloading the dataset

First install the kaggle CLI.

```bash
pip install kaggle
```

Go to your Kaggle Account page and click on `Create new API Token` to download the `~/.kaggle/kaggle.json` file for authentication.

```bash
kaggle datasets download gabrielvanzandycke/deepsport-dataset
unzip -qo ./deepsport-dataset.zip -d deepsport_dataset
```

### Generating COCO-format annotation files

Once the dataset is downloaded and unzipped, the command

```bash
python3 tools/gen_annotations.py
```

should create the COCO-format JSON files for the various splits.

### About the splits

The provided annotations are first split in a *trainval* set (XXX images) and a *test* set (XXX images), each containing images taken from different arenas. We further split the *trainval* set in the *train* (XXX images) and *val* (XX images) sets.

We encourage to use those sets as it pleases. Another set of unannotated images will be provided later to establish the scoreboard.

To make the split as convenient as possible to use, each of *train*, *val*, *test*, *trainval* and *trainvaltest* sets has its own JSON. It could for instance be useful to train the very final model on *trainvaltest* for it to have seen as much data as possible before inference on the challenge images.

### Installing MMDet

https://mmdetection.readthedocs.io/en/v2.2.0/install.html


```
mim install mmdet==2.21.0
```

## Baselines

### YOLACT

```bash
python tools/train.py TODO/yolact_r50_1x8_keemotion_bn_4x.py --cfg-options optimizer.lr=0.003
```

### Mask-RCNN

```bash
python tools/train.py TODO/mask_rcnn_x101_64x4d_fpn_1x_challenge.py --cfg-options optimizer.lr=0.003
```

## Test, inference and submission

Testing can be performed using the following command:

```bash
python tools/test.py configs/challenge/mask_rcnn_x101_64x4d_fpn_1x_challenge.py \
    work_dirs/mask_rcnn_x101_64x4d_fpn_1x_challenge/epoch_5.pth \
    --cfg-options data.test.ann_file=deepsport_dataset/test.json \
    --show-dir output-vis \
    --out test-output.pkl \
    --eval bbox segm
```

When the challenge set is released (as a new set of images and a `challenge.json` file), the following commands could be used to obtain the submission file:

```bash
python tools/test.py configs/challenge/mask_rcnn_x101_64x4d_fpn_1x_challenge.py \
    work_dirs/mask_rcnn_x101_64x4d_fpn_1x_challenge/epoch_5.pth \
    --cfg-options data.test.ann_file=deepsport_dataset/challenge.json \
    --show-dir output-vis \
    --out challenge-output.pkl
python tools/convert_output.py challenge-output.pkl
```

And here should appear the resulting `challenge-output.json` file ready to be uploaded on EvalAI.

## License