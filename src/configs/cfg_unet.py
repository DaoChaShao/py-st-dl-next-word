#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 15:15
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   cfg_unet.py
# @Desc     :   

from dataclasses import dataclass, field

from src.configs.cfg_base import FilePaths, Database
from src.configs.cfg_base4dl import DataPreprocessor, Hyperparameters


@dataclass
class UNetParams:
    INITIAL_FILTERS: int = 64
    SEG_CLASSES: int = 1  # Binary segmentation: 1 (channel output)


@dataclass
class Configuration4UNet:
    DATABASE: Database = field(default_factory=Database)
    FILEPATHS: FilePaths = field(default_factory=FilePaths)
    HYPERPARAMETERS: Hyperparameters = field(default_factory=Hyperparameters)
    PARAMETERS: UNetParams = field(default_factory=UNetParams)
    PREPROCESSOR: DataPreprocessor = field(default_factory=DataPreprocessor)


CONFIG4UNET = Configuration4UNet()
