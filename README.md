# DeepSportRadar Instance Segmentation Challenge

## TODO

- [ ] Adapt when Kaggle is fixed
- [ ] Lay down the submission format
- [ ] Lay down the dataset layout
- [ ] Provide examples

## Installation

### Downloading the dataset

First install the kaggle CLI.

```bash
pip install kaggle
```

Go to your Kaggle Account page and click on `Create new API Token` to download the file to be saved as `~/.kaggle/kaggle.json` for authentication.

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

We encourage to use those sets as it pleases. Another set of **unannotated** images, the *challenge* set will be provided later to establish the scoreboard.

To make the splits as convenient as possible to use, each of *train*, *val*, *test*, *trainval* and *trainvaltest* sets have their own JSON. It is for instance useful to train the very final model on *trainvaltest* for it to have seen as much data as possible before inference on the *challenge* images.

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
    --show-dir test-vis \
    --out test-output.pkl \
    --eval bbox segm
```

When the challenge set is released (as a new set of images and a `challenge.json` file **without no annotation***), the following commands could be used to obtain the submission file:

```bash
python tools/test.py configs/challenge/mask_rcnn_x101_64x4d_fpn_1x_challenge.py \
    work_dirs/mask_rcnn_x101_64x4d_fpn_1x_challenge/epoch_5.pth \
    --cfg-options data.test.ann_file=deepsport_dataset/challenge.json \
    --show-dir challenge-vis \
    --out challenge-output.pkl
python tools/convert_output.py challenge-output.pkl
```

And here should appear the resulting `challenge-output.json` file ready to be uploaded on EvalAI.

## Participating with another codebase

It is totally possible to use another codebase that this one to participate in the challenge. The dataset images and split files should be usable in any codebase able to read a COCO format dataset. Only specificity is that this one has one class with ID 1 for humans. For compatiblity reasons, MMDet methods are used with two classes, and the class 0 never used.

What really matters in the end is for the submission file to be in the right format: the challenge-output.json should have the following layout:

```
[image_result] A list of image results, in the same order as the images in challenge.json

image_result: [
    [bboxes],
    [masks]
]

bboxes: A list of
```

## License

This repository is built from, and on top of MMDet, and distributed under the Apache 2.0 License.
