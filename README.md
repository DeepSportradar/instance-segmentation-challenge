[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/JvMQgMkpkm)
[![Compete on EvalAI](https://badgen.net/badge/compete%20on/EvalAI/blue)](https://eval.ai/web/challenges/challenge-page/2070/overview)
[![Win 2x$500](https://badgen.net/badge/win/2x%20%24500/yellow)](http://mmsports.multimedia-computing.de/mmsports2023/challenge.html)
[![Kaggle Dataset](https://badgen.net/badge/kaggle/dataset/blue)](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset)

# DeepSportRadar Instance Segmentation Challenge (v2, 2023) <!-- omit in toc -->

**This repository is an improved version of last year's edition. It has been updated to work with MMDet v3 and is based on a novel instance segmentation metric targetting occlusions. [More information here](#updates-with-respect-to-last-year-edition)**.

One of the [ACM MMSports 2023 Workshop](http://mmsports.multimedia-computing.de/mmsports2023/index.html) challenges. An opportunity to publish, as well as a $1000 prize by competing on [EvalAI](https://eval.ai/web/challenges/challenge-page/2070/overview). See [this page](http://mmsports.multimedia-computing.de/mmsports2023/challenge.html) for more details.

Congratulations again to the [2022 winners](#2022-winners)! Please do not hesitate to reuse their [code](https://github.com/YJingyu/Instanc_Segmentation_Pro) or [ideas](https://arxiv.org/abs/2209.13899) for this edition.

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
- [Updates with respect to last year edition](#updates-with-respect-to-last-year-edition)
  - [The Occlusion Metric](#the-occlusion-metric)
- [2022 Winners](#2022-winners)
- [Citation](#citation)
- [License](#license)

This challenge tackles the segmentation of individual humans (players, coaches and referees) on a basketball court. We believe the main characteristics of this dataset are that

- players are highly deformable, with some thin parts (arms, legs);
- there is a high level of occlusions between players, and reconnection through occlusions is something we may want our models to perform;
- the amount of data is quite limited compared to some other instance segmentation datasets.

Futhermore, the fact that humans are approximately the same size makes the metrics less tricky to break down, to focus on those particular problems.

This year, the focus will be put on solving occlusions.

## Challenge rules

1. As this is a segmentation challenge, the goal is to obtain the highest **occlusion metric** ([described here](https://github.com/DeepSportradar/instance-segmentation-challenge#the-occlusion-metric)) on images that were not seen during training. In particular, the leaderboards that provide rewards will be built on the unannotated *challenge* set.

2. Only the data and annotations provided by this challenge can be used for training the model. We however accept that the initial weigths of part, or the complete network, come from an established model. (exact source has to be provided in the report/paper)

3. Participants are allowed to train their final model on all provided data (train + test sets) before evaluating on the challenge set.

4. Annotations provided should not be modified, and no annotation should be added. We propose to discuss erroneous annotations on Discord and to integrate corrections for the next version of the challenge.

5. Any team of one or more members can participate in the challenge, except the challenge organizers. However, only one account per team should be used for submission on the challenge set so as to have a consistent limit for every team.

6. After the first phase of the challenge, a publication in the workshop proceedings will be conditioned to the acceptance through the peer-review process and registration to the workshop.

The complete set of rules is available on the EvalAI [challenge evaluation page](https://eval.ai/web/challenges/challenge-page/2070/evaluation).

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

The *test* split should be used to communicate about your model performance publicly, and your model should never see it during training, except maybe for a final submission on the challenge. An **unannotated** set of images, the *challenge* set is provided to establish the true challenge leaderboards.

To make the splits as convenient as possible to use, each of *train*, *val*, *test*, *trainval* and *trainvaltest* sets have their own JSON. It could for instance be useful to train the very final model on *trainvaltest* for it to have seen as much data as possible before inference on the *challenge* images.

## Using MMDet

### Installation

For simplicity, as recommended in [MMDet's documentation](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation), we propose to install MMCV and MMEngine using MIM, and MMDet from github:

```bash
pip3 install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# EITHER
mim install mmdet
# OR
git clone https://github.com/open-mmlab/mmdetection.git
pip install -v -e mmdetection
```

### Baseline

We propose the well-established Mask-RCNN model provided by MMDet as a baseline for this challenge.

```bash
python3 tools/train.py configs/challenge/mask_rcnn_r50_fpn_200e_challenge.py
```

Feel free to integrate any improvement or try a completely different model! Bottom-up or transformer-based methods are of course allowed.

### Test, metrics and submission

Testing can be performed using the following command:

```bash
python3 tools/test.py configs/challenge/mask_rcnn_r50_fpn_200e_challenge.py \
    work_dirs/mask_rcnn_r50_fpn_200e_challenge/epoch_200.pth \
    --ann-file annotations/test.json\
    --show-dir test-vis \
    --out test-output.json
```

For generating a submission for the challenge set (the set of images and a `challenge.json` file **without any annotation**), the following commands can be used to obtain the submission file:

```bash
python3 tools/test.py configs/challenge/mask_rcnn_r50_fpn_200e_challenge.py \
    work_dirs/mask_rcnn_r50_fpn_200e_challenge/epoch_200.pth \
    --ann-file annotations/challenge.json \
    --show-dir challenge-vis \
    --out challenge-output.json
```

And here should appear the resulting `challenge-output.json` file ready to be uploaded on [EvalAI](https://eval.ai/web/challenges/challenge-page/2070/overview). Please note that the output metrics computed locally do not make sense as there are no annotations for that set.

## Participating with another codebase

It is totally possible to use another codebase than this one to participate in the challenge. The dataset images and split files should be usable in any codebase able to read a COCO format dataset. Only specificity is that this one has one class with ID 0 for humans. If needed, MMDet methods could be used with two classes, and the class 1 never used.

### Submission format

What really matters in the end is for the submission file to be in the right format: the challenge-output.json should have the following layout:

```
[image_result] A list of image results, in the same order as the images
               in the JSON set file

image_result: {
    "labels": [class],    the label for each detection (should be 0)
    "scores": [score],    a confidence for each detection (between 0 and 1)
    "bboxes": [bbox],     one bounding box for each detection
    "masks":  [rle_mask]  one rle-encoded mask for each detection
}

bbox: [x1, y1, x2, y2]

rle_mask: {
    "size": [H, W], the mask shape, basically image height and width
    "counts": string with RLE encoding of the mask, along the lines of
              mask_tools.encode(np.asfortranarray(mask))["counts"].decode('utf-8')
}
```

More details to generate the RLE representation from masks can be found in [tools/gen_annotations.py](tools/gen_annotations.py#L47=). Bounding boxes can also be computed from the mask as is demonstrated [there](tools/gen_annotations.py#L54=), please don't forget to reorder the items if needed.

### Computing metrics

Metrics with respect to the test set (or any other set) can be computed using the `tools/test_json.py` script.

```bash
python3 tools/test_json.py --evaluate test-output.json \
    --gt annotations/test.json
```

Alternatively, submitting to the [`Test` phase on EvalAI](https://eval.ai/web/challenges/challenge-page/2070/phases) will provide the same results.

## Updates with respect to last year edition

There are two main changes to this year's edition:

- A novel metric is introduced, that focuses on the resolution of occlusions.
- The repository is using MMDet v3, so as to have easy access to the latest open-source models.

### The Occlusion Metric

The metric determining the winner for this challenge will only take into account instances that appear split because of occlusions (according to ground-truth annotations). The assumption behind this choice is that, given the nature of occlusions, models that are able to solve occluded instances should naturally solve easier ones. Therefore we choose to focus on the hardest difficulties of occlusions: reconnecting disconnected pixels. Whether this assumption holds is actually a good question, as well as whether the proposed metric exploits it sufficiently well. We invite you to prove us wrong by reaching a high Occlusion Metric (OM) and a low mAP. Reaching a high mAP does, however, not induce any penalty: we are equally happy if the obtained models are good for real.

The OM metric is the product of Occluded Instance Recall (OIR), *i.e. the recall of instances appearing visually split*, and the  Disconnected Pixel Recall (DPR), i.e. *the recall of pixels disconnected from the main body of recalled split instances*.

For a ground-truth instance, its true positive (TP) prediction is that with the highest IoU (and above 0.5), if any. All the others are false positives (FP) predictions. Ground-truth instances without associated prediction are false negatives (FN). In order to penalize FP instances in the metric, the contribution of a disconnected pixel recalled by a TP prediction to the DPR is lowered by other FPs that include it.

```
# OM Computation Pseudocode

Given ground-truth instance masks
  and predicted instance masks
  and predicted instance scores

Compute the IoU between ground-truth and predicted instance masks
Assign, to each ground-truth mask, the predicted mask with highest
  IoU ( > 0.5 )
  => yielding TP, FP and FN

Keep only TP and FN instances whose annotated mask is made of
  several connected components
Compute OIR = |TP|/(|TP|+|FN|)

For each TP instance:
  Ignore the largest connected component of the annotated mask
  For each pixel in other annotated connected components:
    Compute reward = current instance score / sum of scores of
      instances predicting this pixel
    If pixel in the predicted mask:
      R += reward
    T += 1
Compute DPR = R/T

OM = DPR*OIR
```

## 2022 Winners

Congrats again to the *Bo Yan, Fengliang Qi, Zhuang Li, Yadong Li, Hongbin Wang* from the *Ant Group* for winning the 2022 edition. Their code is available [in this repository](https://github.com/YJingyu/Instanc_Segmentation_Pro) ([mirror](https://github.com/DeepSportradar/2022-winners-instance-segmentation-challenge)). They presented the ideas behind their method in [this report](https://arxiv.org/abs/2209.13899).

## Citation

If you use any DeepSportradar dataset in your research or wish to refer to the baseline results and discussion published in [our paper](https://arxiv.org/abs/2208.08190), please use the following BibTeX entry:

```
@inproceedings{Van_Zandycke_2022,
  author = {Gabriel Van Zandycke and Vladimir Somers and Maxime Istasse and Carlo Del Don and Davide Zambrano},
  title = {{DeepSportradar}-v1: Computer Vision Dataset for Sports Understanding with High Quality Annotations},
  booktitle = {Proceedings of the 5th International {ACM} Workshop on Multimedia Content Analysis in Sports},
  publisher = {{ACM}},
  year = 2022,
  month = {oct},
  doi = {10.1145/3552437.3555699},
  url = {https://doi.org/10.1145%2F3552437.3555699}
}
```

## License

- This repository is built from, and on top of [MMDet](https://github.com/open-mmlab/mmdetection), and distributed under the [Apache 2.0 License](https://github.com/DeepSportradar/instance-segmentation-challenge/blob/master/LICENSE).
- The challenge data, hosted on [Kaggle](https://www.kaggle.com/datasets/gabrielvanzandycke/deepsport-dataset), is available under the [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) license.
