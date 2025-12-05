#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/5 20:17
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Tools for Next Word Prediction Streamlit Application
----------------------------------------------------------------
Provides page configuration and multi-page navigation utilities.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.2.0"

from .layout import config_page, set_pages

__all__ = [
    "config_page",
    "set_pages"
]
