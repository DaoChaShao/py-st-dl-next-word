#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 14:56
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   cfg_base4dl.py
# @Desc     :   

from dataclasses import dataclass, field
from torch import cuda


@dataclass
class DataPreprocessor:
    BATCHES: int = 16
    DROPOUT: float = 0.3
    IMAGE_HEIGHT: int = 320
    IMAGE_WIDTH: int = 384
    PCA_VARIANCE_THRESHOLD: float = 0.95
    RANDOMNESS: int = 27
    SHUFFLE: bool = True
    TEST_SIZE: float = 0.2


@dataclass
class Hyperparameters:
    ACCELERATOR: str = "cuda" if cuda.is_available() else "cpu"
    # ALPHA: float = 1e-4
    DECAY: float = 1e-4
    # EPOCHS: int = 100


@dataclass
class Config4DL:
    HYPERPARAMETERS: Hyperparameters = field(default_factory=Hyperparameters)
    PREPROCESSOR: DataPreprocessor = field(default_factory=DataPreprocessor)


CONFIG4DL = Config4DL()
