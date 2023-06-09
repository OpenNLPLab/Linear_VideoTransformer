# ------------------------------------------------------------------------
# Mostly a modified copy from timm (https://github.com/facebookresearch/SlowFast)
# ------------------------------------------------------------------------

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import io
import h5py
import numpy as np
import slowfast.utils.logging as logging
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager
from PIL import Image
from petrel_client.client import Client
from . import decoder as decoder
from . import image_decoder as image_decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .data_augment import (
    augment_and_mix_transform,
    auto_augment_transform,
    rand_augment_transform,
)
from .rand_erasing import RandomErasing
from .transform import color_jitter, horizontal_flip, random_scale_and_resize

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
            self.use_hdf5 = True
            self.use_ceph = True
        elif self.mode in ["test"]:
            self.use_hdf5 = True
            self.use_ceph = True
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Kinetics {}...".format(mode))

        if self.use_hdf5:
            self._construct_loader_hdf5()
        else:
            self._construct_loader()
        self._labels = torch.tensor(self._labels)
        self._spatial_temporal_idx = torch.tensor(self._spatial_temporal_idx)
        self._construct_augmentations()

        self.mclient = Client(enable_mc=True)

    def _construct_augmentations(self):
        self.random_erasing = (
            RandomErasing(probability=self.cfg.DATA.RAND_CROP, syncronized=True)
            if self.cfg.DATA.RAND_CROP > 0.0
            else None
        )
        if self.cfg.DATA.AUTOAUGMENT:
            self.auto_augment = auto_augment_transform(
                "v0", dict(translate_const=int(224 * 0.45))
            )
        else:
            self.auto_augment = None

        if self.cfg.DATA.RANDAUGMENT:
            self.rand_augment = rand_augment_transform(
                "rand-m20-n2", dict(translate_const=int(224 * 0.45))
            )
        else:
            self.rand_augment = None

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (
                    len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                    == 2
                )
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                
                ########################process sunweixuan's annotations#######################
                prefix = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR + 'test/')
                filename = os.path.basename(path)
                path = os.path.join(prefix, filename)
                ###############################################################################
                
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def _construct_loader_hdf5(self):
        """
        Construct the video loader from hdf5.
        """
        mode_name = "train" if self.mode == "train" else "val"
        data_file_name = f"{mode_name}.txt"
        fps_data_file_name = f"{mode_name}_fps.txt"
        path_prefix = f"seq_h5/{mode_name}"
        if not os.path.isdir(
            os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, path_prefix)
        ):
            path_prefix = mode_name

        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "annotations", data_file_name
        )
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )
        path_to_fps_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "annotation", fps_data_file_name
        )
        if not PathManager.exists(path_to_fps_file):
            print(
                "{} dir not found, fps will not be used".format(
                    path_to_fps_file
                )
            )
            fps_data_dict = None
        else:
            fps_data_dict = {}
            with PathManager.open(path_to_fps_file, "r") as f:
                for path_fps in f.read().splitlines():
                    assert (
                        len(path_fps.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                        == 2
                    )
                    path, fps = path_fps.split(
                        self.cfg.DATA.PATH_LABEL_SEPARATOR
                    )
                    fps_data_dict[path] = float(fps)
            logger.info(f"Parsed {len(fps_data_dict.keys())} meta_fps")

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        num_skipped_videos = 0
        with PathManager.open(path_to_file, "r") as f:
            clip_idx = 0
            for path_label in f.read().splitlines():
                assert (
                    len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                    == 3
                )
                path, num_frames, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                
                ########use weixuan's annatation################
                path = path.split('h5/')[-1].split('.h5')[0]
                ###########################################
                
                num_frames = int(num_frames)
                if num_frames < 3:
                    num_skipped_videos += 1
                    continue
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(
                            self.cfg.DATA.PATH_TO_DATA_DIR, path_prefix, path
                        )
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {
                        "num_frames": num_frames
                    }
                    if fps_data_dict is not None:
                        self._video_meta[clip_idx * self._num_clips + idx][
                            "fps"
                        ] = fps_data_dict[path]
                clip_idx += 1
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )
        logger.info(f"Number of videos skipeed: {num_skipped_videos}")

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE

        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = self.cfg.DATA.SAMPLING_RATE

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            # Decode from hdf5
            if self.use_hdf5:
                frames = None
                if self.use_ceph:
                    h5_file_name = self._path_to_videos[index].split('/')[-1]
                    if 'k400' in self.cfg.DATA.PATH_TO_DATA_DIR:
                        value = self.mclient.Get(f's3://mmg_data_cv/kinetics_400_h5/{h5_file_name}.h5')
                    elif 'k600' in self.cfg.DATA.PATH_TO_DATA_DIR:
                        value = self.mclient.Get(f's3://mmg_data_cv/kinetics_600_h5/{h5_file_name}.h5')
                    value_buf = io.BytesIO(value)
                    with h5py.File(value_buf) as video_h5:
                        video_key = list(video_h5.keys())[0]

                        frames, frames_index = image_decoder.decode(
                            video_h5,
                            video_key,
                            sampling_rate,
                            self.cfg.DATA.NUM_FRAMES,
                            temporal_sample_index,
                            self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                            video_meta=self._video_meta[index],
                            target_fps=self.cfg.DATA.TARGET_FPS,
                            max_spatial_scale=max_scale,
                            mode=self.mode,
                        )
                else:
                    with h5py.File(
                        f"{self._path_to_videos[index]}.h5", "r"
                    ) as video_h5:
                        video_key = list(video_h5.keys())[0]

                        frames, frames_index = image_decoder.decode(
                            video_h5,
                            video_key,
                            sampling_rate,
                            self.cfg.DATA.NUM_FRAMES,
                            temporal_sample_index,
                            self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                            video_meta=self._video_meta[index],
                            target_fps=self.cfg.DATA.TARGET_FPS,
                            max_spatial_scale=max_scale,
                            mode=self.mode,
                        )

                if frames is None:
                    continue
            else:
                video_container = None
                try:
                    video_container = container.get_video_container(
                        self._path_to_videos[index],
                        self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                        self.cfg.DATA.DECODING_BACKEND,
                    )
                    # print('loaded!!!!!')
                except Exception as e:
                    logger.info(
                        "Failed to load video from {} with error {}".format(
                            self._path_to_videos[index], e
                        )
                    )
                # Select a random video if the current video was not able to access.
                if video_container is None:
                    logger.warning(
                        "Failed to meta load video idx {} from {}; trial {}(video container none)".format(
                            index, self._path_to_videos[index], i_try
                        )
                    )
                    if (
                        self.mode not in ["test"]
                        and i_try > self._num_retries // 2
                    ):
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue

                # Decode video. Meta info is used to perform selective decoding.
                try:
                    frames, frames_index = decoder.decode(
                        video_container,
                        sampling_rate,
                        self.cfg.DATA.NUM_FRAMES,
                        temporal_sample_index,
                        self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                        video_meta=self._video_meta[index],
                        target_fps=self.cfg.DATA.TARGET_FPS,
                        backend=self.cfg.DATA.DECODING_BACKEND,
                        max_spatial_scale=min_scale,
                    )
                except:
                    
                    index = random.randint(0, len(self._path_to_videos) - 1)
                    continue


            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}(frames none)".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            if self.mode == "train":
                if (
                    self.auto_augment is not None
                    or self.rand_augment is not None
                ):
                    if isinstance(frames, torch.Tensor):
                        frames = torch.split(frames, 1, dim=0)
                        frames = [b.squeeze(0).numpy() for b in frames]

                    frames = [
                        Image.fromarray(fmr.astype("uint8")) for fmr in frames
                    ]
                    # print('length:', len(frames))
                if self.auto_augment is not None:
                    frames = self.auto_augment(frames)
                elif self.rand_augment is not None:
                    frames = self.rand_augment(frames)

                if (
                    self.auto_augment is not None
                    or self.rand_augment is not None
                ):
                    frames = torch.as_tensor(np.stack(frames))

            if (
                self.cfg.DATA.USE_XVIT_AGUMENTATION
                and spatial_sample_index == -1
            ):
                frames = random_scale_and_resize(
                    frames,
                    (
                        self.cfg.DATA.TRAIN_CROP_SIZE,
                        self.cfg.DATA.TRAIN_CROP_SIZE,
                    ),
                )

            # color augmentation
            if self.cfg.DATA.COLORAUGMENT:
                frames = frames.permute(0, 3, 1, 2).contiguous().float() / 255.0
                frames = color_jitter(
                    frames,
                    img_brightness=0.8,
                    img_contrast=0.8,
                    img_saturation=0.8,
                )
                frames = frames.permute(0, 2, 3, 1).contiguous() * 255.0

            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )

            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)

            if (
                self.cfg.DATA.USE_XVIT_AGUMENTATION
                and self.cfg.DATA.RANDOM_FLIP
            ):
                if spatial_sample_index == -1:
                    frames, _, _ = horizontal_flip(0.5, frames)

            if (
                not self.cfg.DATA.USE_XVIT_AGUMENTATION
                or spatial_sample_index != -1
            ):
                # Perform data augmentation.
                frames = utils.spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )

            if self.mode == "train":
                if self.random_erasing is not None:
                    frames = self.random_erasing(frames)

            label = self._labels[index]
            frames = utils.pack_pathway_output(self.cfg, frames, frames_index)
            # print('frames shape:', frames[0].shape)
            file_name = self._path_to_videos[index].split('/')[-1]
            return frames, label, index, {}, file_name
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
