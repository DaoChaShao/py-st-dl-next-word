#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 14:52
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   cfg_rnn.py
# @Desc     :   

from dataclasses import dataclass, field

from src.configs.cfg_base import FilePaths
from src.configs.cfg_base4dl import DataPreprocessor, Hyperparameters


@dataclass
class RNNParams:
    CLASSES: int = 2  # Binary classification is 2
    EMBEDDING_DIM: int = 256
    HIDDEN_SIZE: int = 64
    LAYERS: int = 2
    TEMPERATURE: float = 1.0


@dataclass
class Configuration4RNN:
    FILEPATHS: FilePaths = field(default_factory=FilePaths)
    HYPERPARAMETERS: Hyperparameters = field(default_factory=Hyperparameters)
    PARAMETERS: RNNParams = field(default_factory=RNNParams)
    PREPROCESSOR: DataPreprocessor = field(default_factory=DataPreprocessor)


CONFIG4RNN = Configuration4RNN()
