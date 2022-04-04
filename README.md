# CHALLENGE STARTS OFFICIALLY APRIL 6th. Stay tuned. <!-- omit in toc -->

[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/JvMQgMkpkm) [![Compete on EvalAI](https://badgen.net/badge/compete%20on/EvalAI/blue)](https://eval.ai/web/challenges/challenge-page/1685/overview) [![Win $1,000.00](https://badgen.net/badge/win/%241%2C000.00/yellow)](http://mmsports.multimedia-computing.de/mmsports2022/challenge.html)

# DeepSportRadar Instance Segmentation Challenge <!-- omit in toc -->

One of the [ACM MMSports 2022 Workshop](http://mmsports.multimedia-computing.de/mmsports2022/index.html) challenges. An opportunity to publish, as well as a $1000,00 prize. See [this page](http://mmsports.multimedia-computing.de/mmsports2022/challenge.html) for more details.

**Table of contents**
- [Installation](#installation)
  - [Downloading the dataset](#downloading-the-dataset)
  - [The COCO-format annotation files](#the-coco-format-annotation-files)
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

The dataset can be found [here](https://www.kaggle.com/datasets/gabrielvanzandycke/deepsport-dataset). It can be downloaded and unzipped manually in the deepsport_dataset folder of the project.

We will here download it programmatically. First install the kaggle CLI.

```bash
pip install kaggle
```

Go to your Kaggle Account page and click on `Create new API Token` to download the file to be saved as `~/.kaggle/kaggle.json` for authentication.

```bash
kaggle datasets download gabrielvanzandycke/deepsport-dataset
unzip -qo ./deepsport-dataset.zip -d deepsport_dataset
```

### The COCO-format annotation files

The `.json` files provided in the `annotations/` folder by this repository are annotation files for the different splits of the dataset. They comply with the COCO format, and can be re-generated using the following command once the dataset is setup:

```bash
python3 tools/gen_annotations.py
# train 211 images 1606 annotations
# val 35 images 293 annotations
# trainval 246 images 1899 annotations
# test 64 images 477 annotations
# trainvaltest 310 images 2376 annotations
```

Having no change with respect to the annotation files distributed confirms that the dataset is properly setup.

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

When the challenge set is released (as a new set of images and a `challenge.json` file **without no annotation**), the following commands could be used to obtain the submission file:

```bash
python3 tools/test.py configs/challenge/mask_rcnn_x101_64x4d_fpn_20e_challenge.py \
    work_dirs/mask_rcnn_x101_64x4d_fpn_20e_challenge/latest.pth \
    --cfg-options data.test.ann_file=annotations/challenge.json \
    --show-dir challenge-vis \
    --out challenge-output.pkl
python3 tools/convert_output.py challenge-output.pkl
```

And here should appear the resulting `challenge-output.json` file ready to be uploaded on [EvalAI](https://eval.ai/web/challenges/challenge-page/1685/overview).

## Participating with another codebase

It is totally possible to use another codebase that this one to participate in the challenge. The dataset images and split files should be usable in any codebase able to read a COCO format dataset. Only specificity is that this one has one class with ID 1 for humans. For compatiblity reasons, MMDet methods are used with two classes, and the class 0 never used.

### Submission format

What really matters in the end is for the submission file to be in the right format: the challenge-output.json should have the following layout:


```
[image_result] A list of image results, in the same order as the images
               in the JSON set file

image_result: [
    [bbox],     one bounding box for each detection
    [rle_mask]  one rle-encoded mask for each detection
]

bbox: [x1, y1, x2, y2, confidence]

rle_mask: {
    "size": [H, W], the mask shape, basically image height and width
    "counts": string with RLE encoding of the mask, along the lines of
              mask_tools.encode(np.asfortranarray(mask))["counts"].decode('utf-8')
}
```

More details to generate the RLE representation from masks can be found in [tools/gen_annotations.py](tools/gen_annotations.py#L47=). Bounding boxes can also be computed from the mask as is demonstrated [there](tools/gen_annotations.py#L54=), please don't forget to add the confidence.

### Computing metrics

Metrics with respect to the test set (or any other set) can be computed using the `tools/test_json.py` script.

```bash
python3 tools/test_json.py test-output.json \
    --cfg-options data.test.ann_file=annotations/test.json
```

## License

This repository is built from, and on top of MMDet, and distributed under the Apache 2.0 License.
