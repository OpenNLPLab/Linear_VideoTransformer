#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import faulthandler
faulthandler.enable()

import sys
import utils
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
sys.path.append(".")

from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
from test_net import test
from train_net import train
from visualization import visualize

"""Wrapper to train and test a video classification model."""


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    is_init = False
    if cfg.TRAIN.ENABLE:
        train(cfg)
        is_init = True
    if cfg.TEST.ENABLE:
        test(cfg, is_init=is_init)
    if cfg.TENSORBOARD.ENABLE and (
        cfg.TENSORBOARD.MODEL_VIS.ENABLE
        or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    ):
        visualize(cfg)
    # Perform training.
    # if cfg.TRAIN.ENABLE:
    #     launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    # if cfg.TEST.ENABLE:
    #     launch_job(cfg=cfg, init_method=args.init_method, func=test)

    # Perform model visualization.
    # if cfg.TENSORBOARD.ENABLE and (
    #     cfg.TENSORBOARD.MODEL_VIS.ENABLE
    #     or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    # ):
    #     launch_job(cfg=cfg, init_method=args.init_method, func=visualize)


if __name__ == "__main__":
    main()
