#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from demo_net import demo
from test_net import test
from train_net import train
from visualization import visualize
import torch


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)

        # extra setting for oob
        cfg.TRAIN.ENABLE = False
        cfg.LOG_MODEL_INFO = False
        cfg.NUM_GPUS = 0
        cfg.DATA.DECODING_BACKEND = 'pyav'
        cfg.DATA.PATH_PREFIX = args.dataset_dir
        cfg.DATA.PATH_LABEL_SEPARATOR = ','
        cfg.TEST.BATCH_SIZE = args.batch_size
        if cfg.triton_cpu:
            print("run with triton cpu backend")
            import torch._inductor.config
            torch._inductor.config.cpu_backend="triton"
        # Perform training.
        if cfg.TRAIN.ENABLE:
            launch_job(cfg=cfg, init_method=args.init_method, func=train)

        # Perform multi-clip testing.
        import torch
        if cfg.TEST.ENABLE:
            if cfg.precision == "bfloat16":
                print("---- Use AMP bfloat16")
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.bfloat16):
                    launch_job(cfg=cfg, init_method=args.init_method, func=test)
            elif cfg.precision == "float16":
                print("---- Use AMP float16")
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.half):
                    launch_job(cfg=cfg, init_method=args.init_method, func=test)
            else:
                launch_job(cfg=cfg, init_method=args.init_method, func=test)

        # Perform model visualization.
        if cfg.TENSORBOARD.ENABLE and (
            cfg.TENSORBOARD.MODEL_VIS.ENABLE
            or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
        ):
            launch_job(cfg=cfg, init_method=args.init_method, func=visualize)

        # Run demo.
        if cfg.DEMO.ENABLE:
            demo(cfg)


if __name__ == "__main__":
    main()
