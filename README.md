[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/JvMQgMkpkm)
[![Compete on EvalAI](https://badgen.net/badge/compete%20on/EvalAI/blue)](https://eval.ai/web/challenges/challenge-page/1685/overview)
[![Win 2x $500](https://badgen.net/badge/win/2x%20%24500/yellow)](http://mmsports.multimedia-computing.de/mmsports2022/challenge.html)
[![Kaggle Dataset](https://badgen.net/badge/kaggle/dataset/blue)](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset)

# DeepSportRadar Instance Segmentation Challenge <!-- omit in toc -->

## ** The DeepSportRadar Challenges will come back next year with some improvements, stay tuned on our [Discord channel](https://discord.gg/JvMQgMkpkm)! **

One of the [ACM MMSports 2022 Workshop](http://mmsports.multimedia-computing.de/mmsports2022/index.html) challenges. An opportunity to publish, as well as a $1000 prize by competing on [EvalAI](https://eval.ai/web/challenges/challenge-page/1685/overview). See [this page](http://mmsports.multimedia-computing.de/mmsports2022/challenge.html) for more details.

![Instance segmentation banner](https://raw.githubusercontent.com/DeepSportRadar/instance-segmentation-challenge/master/assets/banner_large.png)

**Table of contents**
- [Challenge rules](#challenge-rules)
- [Installation](#installation)
  - [Downloading the dataset](#downloading-the-dataset)
    - [Setting up the challenge set](#setting-up-the-challenge-set)
  - [The COCO-format annotation files](#the-coco-format-annotation-files)
  - [About the splits](#about-the-splits)
- [Using MMDet](#using-mmdet)
  - [Installation](#installation-1)
  - [Baseline](#baseline)
  - [Test, metrics and submission](#test-metrics-and-submission)
- [Participating with another codebase](#participating-with-another-codebase)
  - [Submission format](#submission-format)
  - [Computing metrics](#computing-metrics)
- [Citation](#citation)
- [License](#license)

This challenge tackles the segmentation of individual humans (players, coaches and referees) on a basketball court. We believe the main characteristics of this dataset are that

- players are highly deformable, with some thin parts (arms, legs);
- there is a high level of occlusions between players, and reconnection through occlusions is something we may want our models to perform;
- the amount of data is quite limited compared to some other instance segmentation datasets.

Futhermore, the fact that humans are approximately the same size makes the metrics less tricky to break down, to focus on those particular problems.

## Challenge rules

As this is a segmentation challenge, the goal is to obtain the best `segm_mAP` metric on images that were not seen during training. In particular, the leaderboards that provide rewards will be built on an unannotated *challenge* set that will be provided late in June.

Only the data provided along with this challenge can be used for training the model. We however accept that the initial weigths of part, or the complete network, come from a established model zoo. (exact location has to be provided in the report/paper)

The complete set of rules is available on the EvalAI [challenge evaluation page](https://eval.ai/web/challenges/challenge-page/1685/evaluation).

## Installation

### Downloading the dataset

The dataset can be found [here](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset). It can be downloaded and unzipped manually in the `basketball-instants-dataset/` folder of the project.

We will here download it programmatically. First install the kaggle CLI.

```bash
pip install kaggle
```

Go to your Kaggle Account page and click on `Create new API Token` to download the file to be saved as `~/.kaggle/kaggle.json` for authentication.

```bash
kaggle datasets download deepsportradar/basketball-instants-dataset
unzip -qo ./basketball-instants-dataset.zip -d basketball-instants-dataset
```

#### Setting up the challenge set

```bash
wget https://arena-data.keemotion.com/tmp/gva/mmsports_challenge_set_data.zip
unzip -qo mmsports_challenge_set_data.zip -d basketball-instants-dataset
```

### The COCO-format annotation files

The `.json` files provided in the `annotations/` folder by this repository are annotation files for the different splits of the dataset. They comply with the COCO format, and can be re-generated using the following command once the dataset is setup:

```bash
python3 tools/gen_annotations.py
# train 223 images 1674 annotations
# val 37 images 344 annotations
# trainval 260 images 2018 annotations
# test 64 images 477 annotations
# trainvaltest 324 images 2495 annotations
# challenge 84 images 0 annotations
```

Having no change with respect to the annotation files distributed confirms that the dataset is properly setup.

### About the splits

The provided annotations are first split in a *trainval* set (246 images) and a *test* set (64 images), each containing images taken from different arenas. We further split the *trainval* set in the *train* (211 images) and *val* (35 images) sets in a deterministic manner.

The *test* split should be used to communicate about your model performance publicly, and your model should never see it during training, except maybe for a final submission on the challenge. An **unannotated** set of images, the *challenge* set will be provided later to establish the true challenge leaderboards.

To make the splits as convenient as possible to use, each of *train*, *val*, *test*, *trainval* and *trainvaltest* sets have their own JSON. It could for instance be useful to train the very final model on *trainvaltest* for it to have seen as much data as possible before inference on the *challenge* images.

## Using MMDet

### Installation

For simplicity, we propose to install MMDet using [MIM](https://github.com/open-mmlab/mim).

```bash
pip3 install openmim
mim install mmdet==2.21.0
```

### Baseline

We propose the well-established Mask-RCNN model provided by MMDet as a baseline for this challenge.

```bash
python3 tools/train.py configs/challenge/mask_rcnn_x101_64x4d_fpn_20e_challenge.py
```

Feel free to integrate any improvement or try a completely different model!

### Test, metrics and submission

Testing can be performed using the following command:

```bash
python3 tools/test.py configs/challenge/mask_rcnn_x101_64x4d_fpn_20e_challenge.py \
    work_dirs/mask_rcnn_x101_64x4d_fpn_20e_challenge/latest.pth \
    --show-dir test-vis \
    --out test-output.pkl \
    --eval bbox segm
```

When the challenge set is released (as a new set of images and a `challenge.json` file **without any annotation**), the following commands could be used to obtain the submission file:

```bash
python3 tools/test.py configs/challenge/mask_rcnn_x101_64x4d_fpn_20e_challenge.py \
    work_dirs/mask_rcnn_x101_64x4d_fpn_20e_challenge/latest.pth \
    --cfg-options data.test.ann_file=annotations/challenge.json \
    --show-dir challenge-vis \
    --out challenge-output.pkl
python3 tools/convert_output.py challenge-output.pkl
```

And here should appear the resulting `challenge-output.json` file ready to be uploaded on [EvalAI](https://eval.ai/web/challenges/challenge-page/1685/overview). Please note that it would not make sense to pass the `--eval bbox segm` arguments as there will be no annotation on that set.

## Participating with another codebase

It is totally possible to use another codebase than this one to participate in the challenge. The dataset images and split files should be usable in any codebase able to read a COCO format dataset. Only specificity is that this one has one class with ID 0 for humans. If needed, MMDet methods could be used with two classes, and the class 1 never used.

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

More details to generate the RLE representation from masks can be found in [tools/gen_annotations.py](tools/gen_annotations.py#L47=). Bounding boxes can also be computed from the mask as is demonstrated [there](tools/gen_annotations.py#L54=), please don't forget to add the confidence and reorder the items if needed.

### Computing metrics

Metrics with respect to the test set (or any other set) can be computed using the `tools/test_json.py` script.

```bash
python3 tools/test_json.py test-output.json \
    --cfg-options data.test.ann_file=annotations/test.json
```

Alternatively, submitting to the [`Test` phases on EvalAI](https://eval.ai/web/challenges/challenge-page/1685/phases) will provide the same results.

## Citation

If you use any DeepSportradar dataset in your research or wish to refer to the baseline results and discussion published in [our paper](https://arxiv.org/abs/2208.08190), please use the following BibTeX entry:

    @inproceedings{
    Van_Zandycke_2022,
    author = {Gabriel Van Zandycke and Vladimir Somers and Maxime Istasse and Carlo Del Don and Davide Zambrano},
	title = {{DeepSportradar}-v1: Computer Vision Dataset for Sports Understanding with High Quality Annotations},
	booktitle = {Proceedings of the 5th International {ACM} Workshop on Multimedia Content Analysis in Sports},
	publisher = {{ACM}},
    year = 2022,
	month = {oct},
    doi = {10.1145/3552437.3555699},
    url = {https://doi.org/10.1145%2F3552437.3555699}
    }

## License

This repository is built from, and on top of [MMDet](https://github.com/open-mmlab/mmdetection), and distributed under the Apache 2.0 License.
