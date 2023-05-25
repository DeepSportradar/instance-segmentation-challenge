# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import torch
from mmdet.evaluation import CocoMetric
from occlusion_metric import OcclusionMetric, ann_to_om


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='Test a model')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument(
        '--gt', '--ann-file',
        type=str,
        help='gt file to dump or to evaluate against')
    parser.add_argument(
        '--evaluate',
        type=str,
        help='Only perform evaluation, against given gt file.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    om = OcclusionMetric()
    map = CocoMetric(
        ann_file=args.gt,
        metric=['bbox', 'segm'],
        format_only=False,
        backend_args=None)
    map._dataset_meta = dict(classes=['human'])

    import json
    gts = ann_to_om(args.gt)
    preds = json.load(open(args.evaluate, 'r'))

    for pred in preds:
        pred['bboxes'] = torch.tensor(pred['bboxes'])
        pred['scores'] = torch.tensor(pred['scores'])
        pred['labels'] = torch.tensor(pred['labels'])
        for mask in pred['masks']:
            mask['counts'] = mask['counts'].encode('utf-8')
    results = list(zip(gts, preds))
    print(map.compute_metrics(results))
    print(om.compute_metrics(results))


if __name__ == '__main__':
    main()
