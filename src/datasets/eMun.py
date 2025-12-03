#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 23:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   eMun.py
# @Desc     :   

from enum import StrEnum, unique


@unique
class SeqMode4DataSet(StrEnum):
    SEQ2ONE = "seq2one"
    SEQ2SEQ = "seq2seq"
    SEQ_SLICE = "slice"


if __name__ == "__main__":
    out = SeqMode4DataSet.SEQ2ONE
    print(out)
