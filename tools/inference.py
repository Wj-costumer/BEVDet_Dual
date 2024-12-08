# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
import logging
import torch
import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from tools.misc.fuse_conv_bn import fuse_module

import yaml, argparse, time, os, json, copy, multiprocessing
from tracker.dataloader.nusc_loader import NuScenesloader
from tracker.tracking.nusc_tracker import Tracker
from tracker.data.script.NUSC_CONSTANT import *
from tracker.utils.io import load_file
from typing import List
from tqdm import tqdm
from predictor.converter import get_forecast_data
from predictor.hivt.models.hivt import HiVT

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', default='./ckpts/bev_occ.pth', help='checkpoint file')
    parser.add_argument('--samples', default=500, help='samples to benchmark')
    parser.add_argument('--save_dir', default='./work_dirs/e2e', help='samples to benchmark')
    parser.add_argument('--forecast_ckpt', default='./ckpts/forecaster_128.ckpt', help='samples to benchmark')

    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--no-acceleration',
        action='store_true',
        help='Omit the pre-computation acceleration')
    args = parser.parse_args()
    return args

def track(result_path, token, process, nusc_loader, save=False):
    # PolyMOT modal is completely dependent on the detector modal
    result = {
        "results": {},
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
    }
    # tracking and output file
    nusc_tracker = Tracker(config=nusc_loader.config)
    for frame_data in tqdm(nusc_loader, desc='Running', total=len(nusc_loader) // process, position=token):
        if process > 1 and frame_data['seq_id'] % process != token:
            continue
        sample_token = frame_data['sample_token']
        # track each sequence
        nusc_tracker.tracking(frame_data)
        """
        only for debug
        {
            'np_track_res': np.array, [num, 17] add 'tracking_id', 'seq_id', 'frame_id'
            'box_track_res': np.array[NuscBox], [num,]
            'no_val_track_result': bool
        }
        """
        # output process
        sample_results = []
        if 'no_val_track_result' not in frame_data:
            for predict_box in frame_data['box_track_res']:
                box_result = {
                    "sample_token": sample_token,
                    "translation": [float(predict_box.center[0]), float(predict_box.center[1]),
                                    float(predict_box.center[2])],
                    "size": [float(predict_box.wlh[0]), float(predict_box.wlh[1]), float(predict_box.wlh[2])],
                    "rotation": [float(predict_box.orientation[0]), float(predict_box.orientation[1]),
                                 float(predict_box.orientation[2]), float(predict_box.orientation[3])],
                    "velocity": [float(predict_box.velocity[0]), float(predict_box.velocity[1])],
                    "tracking_id": str(predict_box.tracking_id),
                    "tracking_name": predict_box.name,
                    "tracking_score": predict_box.score,
                }
                sample_results.append(box_result.copy())

        # add to the output file
        if sample_token in result["results"]:
            result["results"][sample_token] = result["results"][sample_token] + sample_results
        else:
            result["results"][sample_token] = sample_results

    # sort track result by the tracking score
    for sample_token in result["results"].keys():
        confs = sorted(
            [
                (-d["tracking_score"], ind)
                for ind, d in enumerate(result["results"][sample_token])
            ]
        )
        result["results"][sample_token] = [
            result["results"][sample_token][ind]
            for _, ind in confs[: min(500, len(confs))]
        ]

    # write file
    if save:
        if process > 1:
            json.dump(result, open(result_path + str(token) + ".json", "w"))
        else:
            json.dump(result, open(result_path + "/results.json", "w"))
    return result

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    if not args.no_acceleration:
        cfg.model.img_view_transformer.accelerate=True
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with several samples and take the average
    det_results = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            det_result = model(return_loss=False, rescale=True, **data)
            det_results.append({'pts_bbox': det_result[0]['pts_bbox']})
    det_results = dataset.format_results(det_results, args.save_dir)
    
    config = yaml.load(open('./tracker/config/nusc_config.yaml', 'r'), Loader=yaml.Loader)
    
    if cfg.data_version == 'v1.0-mini':  
        first_token_path = 'tracker/data/nusc_first_token.json'
    else:
        first_token_path = './tracker/data/nusc_first_token.json'

    nusc_loader = NuScenesloader(args.save_dir + "/pts_bbox/results_nusc.json",
                                 first_token_path,
                                 config)
    track_results = track(args.save_dir, 0, 1, nusc_loader)
    forecast_data = get_forecast_data(cfg.data_version, cfg.data_root, track_results, save=True, save_path=args.save_dir + '/trajectory.json')
    predictor = HiVT.load_from_checkpoint(checkpoint_path=args.forecast_ckpt, parallel=False, strict=False)
    forecast_results = []
    for data in tqdm(forecast_data[:100]):
        result = predictor(data)
        forecast_results.append(result)
    forecast_save_path = os.path.join(args.save_dir, 'forecast_results.pt')
    torch.save(forecast_data, forecast_save_path)
    print(f"Forecast Results have been saved in {forecast_save_path}")

if __name__ == '__main__':
    main()
