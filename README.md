# DeepSportRadar Instance Segmentation Challenge <!-- omit in toc -->

- [ ] Add license file


- [Installation](#installation)
  - [Downloading the dataset](#downloading-the-dataset)
  - [Generating COCO-format annotation files](#generating-coco-format-annotation-files)
  - [About the splits](#about-the-splits)
- [Using MMDet](#using-mmdet)
  - [Installation](#installation-1)
  - [Baseline](#baseline)
  - [Test, metrics and submission](#test-metrics-and-submission)
- [Participating with another codebase](#participating-with-another-codebase)
  - [Submission format](#submission-format)
  - [Computing metrics](#computing-metrics)
- [License](#license)

This challenge tackles the segmentation of individual humans (players, coaches and referees) on a basketball court. We believe the main characteristics of this dataset are that

- players are highly deformable, with some thin parts (arms, legs);
- there is a high level of occlusions between players, and reconnection through occlusions is something we may want our models to perform;
- the amount of data is quite limited compared to some other instance segmentation datasets.

Futhermore, the fact that humans are approximately the same size makes the metrics less tricky to break down, to focus on those particular problems.

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

The provided annotations are first split in a *trainval* set (246 images) and a *test* set (64 images), each containing images taken from different arenas. We further split the *trainval* set in the *train* (211 images) and *val* (35 images) sets in a deterministic manner.

We encourage to use those sets as it pleases. Another set of **unannotated** images, the *challenge* set will be provided later to establish the scoreboard.

To make the splits as convenient as possible to use, each of *train*, *val*, *test*, *trainval* and *trainvaltest* sets have their own JSON. It is for instance useful to train the very final model on *trainvaltest* for it to have seen as much data as possible before inference on the *challenge* images.

## Using MMDet

### Installation

https://mmdetection.readthedocs.io/en/v2.2.0/install.html


```bash
pip install openmim
mim install mmdet==2.21.0
```

### Baseline

We propose the well-established Mask-RCNN model provided by MMDet as a baseline for this challenge.

```bash
python3 tools/train.py configs/challenge/mask_rcnn_x101_64x4d_fpn_20e_challenge.py
```

### Test, metrics and submission

Testing can be performed using the following command:

```bash
python3 tools/test.py configs/challenge/mask_rcnn_x101_64x4d_fpn_20e_challenge.py \
    work_dirs/mask_rcnn_x101_64x4d_fpn_20e_challenge/latest.pth \
    --show-dir test-vis \
    --out test-output.pkl \
    --eval bbox segm
```

When the challenge set is released (as a new set of images and a `challenge.json` file **without no annotation***), the following commands could be used to obtain the submission file:

```bash
python3 tools/test.py configs/challenge/mask_rcnn_x101_64x4d_fpn_20e_challenge.py \
    work_dirs/mask_rcnn_x101_64x4d_fpn_20e_challenge/latest.pth \
    --cfg-options data.test.ann_file=deepsport_dataset/challenge.json \
    --show-dir challenge-vis \
    --out challenge-output.pkl
python3 tools/convert_output.py challenge-output.pkl
```

And here should appear the resulting `challenge-output.json` file ready to be uploaded on EvalAI.

## Participating with another codebase

It is totally possible to use another codebase that this one to participate in the challenge. The dataset images and split files should be usable in any codebase able to read a COCO format dataset. Only specificity is that this one has one class with ID 1 for humans. For compatiblity reasons, MMDet methods are used with two classes, and the class 0 never used.

### Submission format

What really matters in the end is for the submission file to be in the right format: the challenge-output.json should have the following layout:


```
[image_result] A list of image results, in the same order as the images
               in the JSON set file

image_result: [
    [bbox],
    [rle_mask]
]

bbox: [x1, y1, x2, y2, confidence]

rle_mask: {
    "size": [H, W], the mask shape, basically image height and width
    "counts": string with RLE encoding of the mask, along the lines of
              mask_tools.encode(np.asfortranarray(mask)).decode('utf-8')
}
```

More details to generate the RLE representation from masks can be found in [tools/gen_annotations.py](tools/gen_annotations.py#L47=).

### Computing metrics

Metrics with respect to the test set can be computed using the `tools/test_json.py` script.

```bash
python3 tools/test_json.py test-output.json \
    --cfg-options data.test.ann_file=deepsport_dataset/test.json
```

## License

This repository is built from, and on top of MMDet, and distributed under the Apache 2.0 License.
