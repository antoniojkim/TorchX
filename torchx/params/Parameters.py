# -*- coding: utf-8 -*-
import os
import logging
from typing import Dict

import torch

from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class Parameters:
    """
    Args:
        num_epochs: Number of epochs to train for
        epoch_start: Start counting epochs from this number
        batch_size: Number of images in each batch
        checkpoint_step: How often to save checkpoints (epochs)
        validation_step: How often to perform validation (epochs)
        num_validation: How many validation images to use
        num_workers: Number of workers
        learning_rate: learning rate used for training
        cuda: GPU ids used for training
        use_gpu: whether to user gpu for training
        pretrained_model_path: path to pretrained model
        save_model_path: path to save model
        log_file: path to log file
    """

    def __init__(
        self,
        param_file_path: str = None,
        num_epochs: int = 100,
        epoch_start: int = 0,
        batch_size: int = 1,
        checkpoint_step: int = 2,
        validation_step: int = 2,
        num_validation: int = 1000,
        num_workers: int = 1,
        learning_rate: float = 0.001,
        cuda: str = "0",
        use_gpu: bool = True,
        pretrained_model_path: float = None,
        save_model_path: str = "./.checkpoints",
        log_file: str = "./model.log",
        **params
    ):
        self.num_epochs = num_epochs
        self.epoch_start = epoch_start
        self.batch_size = batch_size
        self.checkpoint_step = checkpoint_step
        self.validation_step = validation_step
        self.num_validation = num_validation
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.cuda = cuda
        self.use_gpu = use_gpu
        self.pretrained_model_path = pretrained_model_path
        self.save_model_path = save_model_path
        self.log_file = log_file

        self.__dict__.update(**params)
        if param_file_path is not None and os.path.isfile(param_file_path):
            with open(param_file_path) as file:
                self.__dict__.update(**load(file, Loader=Loader))

        self.use_gpu = self.use_gpu and torch.cuda.is_available()

    def get_logger(self, name: str, overwrite: bool = False, level: int = logging.INFO):
        log = logging.getLogger(name)

        if overwrite:
            with open(self.log_file, "w"):
                pass

        hdlr = logging.FileHandler(self.log_file)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        hdlr.setFormatter(formatter)
        log.addHandler(hdlr)
        log.setLevel(level)
        return log
