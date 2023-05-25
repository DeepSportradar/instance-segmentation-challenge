# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
import collections
import torch, torch.nn as nn, torch.nn.functional as F
from copy import deepcopy
from typing import Sequence, Optional, Any

from mmengine import ConfigDict
from mmengine.fileio import dump
from mmengine.logging import print_log
from mmengine.registry import METRICS
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.evaluator import DumpResults, BaseMetric
from mmengine.evaluator.metric import _to_cpu

from mmdet.structures.mask import encode_mask_results
from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults, CocoMetric
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo
import occlusion_metric


Ñ=None


class DumpPrediction(BaseMetric):
    def __init__(self,
                 out_file_path: str,
                 collect_device: str = 'cpu',
                 collect_dir: Optional[str] = None) -> None:
        super().__init__(
            collect_device=collect_device, collect_dir=collect_dir)
        if not out_file_path.endswith(('.json',)):
            raise ValueError('The output file must be a json file.')
        self.out_file_path = out_file_path

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        data_samples = _to_cpu(data_samples)
        for data_sample in data_samples:
            src = data_sample['pred_instances']
            res = dict()
            res['img_id'] = data_sample['img_id']
            res['bboxes'] = src['bboxes'].tolist()
            res['scores'] = src['scores'].tolist()
            res['labels'] = src['labels'].tolist()
            res['masks'] = encode_mask_results(src['masks'].numpy())
            for m in res['masks']:
                m['counts'] = m['counts'].decode('utf-8')
            self.results.append(res)

    def compute_metrics(self, results: list) -> dict:
        dump(sorted(results, key=lambda res: res['img_id']), self.out_file_path)
        print_log(
            f'Predictions have been saved to {self.out_file_path}.',
            logger='current')
        return {}


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a json file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--ann-file', '--gt', default=None, help='Annotation file')
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
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.ann_file:
        cfg.test_evaluator[0].ann_file = args.ann_file
        cfg.test_dataloader.dataset.ann_file = args.ann_file

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.json')), \
            'The dump file must be a json file.'
        runner.test_evaluator.metrics.append(
            DumpPrediction(out_file_path=args.out))

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
