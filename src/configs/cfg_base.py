#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 14:36
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   cfg_base.py
# @Desc     :   

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass
class FilePaths:
    API_KEY: Path = BASE_DIR / "data/api_keys.yaml"
    DATA4ALL: Path = BASE_DIR / "data/weibo_comments.csv"
    DATA4TRAIN: Path = BASE_DIR / "data/train/"
    DATA4TEST: Path = BASE_DIR / "data/test/"
    DICTIONARY: Path = BASE_DIR / "data/dictionary.json"
    LOGS: Path = BASE_DIR / "logs/"
    SAVED_NET: Path = BASE_DIR / "models/model.pth"
    SPACY_MODEL_EN: Path = BASE_DIR / "models/spacy/en_core_web_md"
    SPACY_MODEL_ZH: Path = BASE_DIR / "models/spacy/zh_core_web_md"


@dataclass
class Config:
    FILEPATHS: FilePaths = field(default_factory=FilePaths)


CONFIG = Config()
