#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:17
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Neural Nets Module - Neural Network Architectures
----------------------------------------------------------------
This module provides a complete set of neural network architectures
for various machine learning tasks including segmentation, classification,
and sequence modeling.

Main Categories:
+ Standard4LayersUNetClassification: 4-layer UNet variant for semantic segmentation
+ Standard5LayersUNetForClassification: 5-layer UNet variant for semantic segmentation
+ LSTMRNNForClassification: Recurrent Neural Network for sequence classification tasks

Usage:
+ Direct import of models via:
    - from src.nn import Standard4LayersUNetClassification, CONFIG4CNN, etc.
+ Instantiate models with default or custom parameters as needed.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.2.0"

from .unet_4layers_sem_seg import Standard4LayersUNetClassification
from .unet_5layers_sem_seg import Standard5LayersUNetForClassification
from .rnn_lstm_classification import LSTMRNNForClassification

__all__ = [
    "Standard4LayersUNetClassification",
    "Standard5LayersUNetForClassification",
    "LSTMRNNForClassification",
]
