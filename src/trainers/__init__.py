#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:19
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Trainers Module - PyTorch Training Implementations
----------------------------------------------------------------
This module provides a complete set of specialized PyTorch trainer
classes for various machine learning tasks including regression,
sequence classification, and semantic segmentation.

Main Categories:
+ TorchTrainer4Regression: Trainer for regression models such as MLP
  or CNN, providing end-to-end training loops and regression metrics

+ TorchTrainer4Seq2Classification: Trainer for sequence classification
  using RNN/GRU/LSTM architectures with full support for sequential
  data batching and classification evaluation

+ TorchTrainer4UNetSemSeg: Trainer for UNet-based semantic segmentation
  with image-mask training cycles, IoU computation, and segmentation
  score evaluation

Usage:
+ Direct import of trainer classes via:
    - from src.trainers import TorchTrainer4Regression, TorchTrainer4UNetSemSeg, etc.
+ Instantiate trainers with model, data loaders, optimizer, and config
  to perform complete supervised training workflows.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.2.0"

from .mlp_regression import TorchTrainer4Regression
from .rnn_seq_classification import TorchTrainer4Seq2Classification
from .unet_sem_seg import TorchTrainer4UNetSemSeg

__all__ = [
    "TorchTrainer4Regression",
    "TorchTrainer4Seq2Classification",
    "TorchTrainer4UNetSemSeg",
]
