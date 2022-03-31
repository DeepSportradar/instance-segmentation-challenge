import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import setup_multi_processes

import pickle
import json
import sys
sys.path.insert(0, './')
from tools.convert_output import json_to_pkl


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('json', help='json output file')
    parser.add_argument('--unsafe', default=False, action='store_true')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    assert args.json.endswith('.json') or args.unsafe

    cfg = Config.fromfile('configs/challenge/deepsportlab_instances.py')
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    if args.json.endswith('.json'):
        outputs = json_to_pkl(json.load(open(args.json, 'r')))
    elif args.json.endswith('.pkl') and args.unsafe:
        outputs = pickle.load(open(args.json, 'rb'))

    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule', 'dynamic_intervals'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=['bbox', 'segm']))
    print(len(dataset))
    metric = dataset.evaluate(outputs, **eval_kwargs)
    print(metric)


if __name__ == '__main__':
    main()
