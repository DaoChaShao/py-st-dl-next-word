#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/4 16:28
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   calc4regression.py
# @Desc     :   

from numpy import ndarray, sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculator_for_regression(outputs: ndarray, targets: ndarray) -> dict[str, float]:
    """ Calculates the mean squared error and mean absolute error
    :param outputs: the predicted outputs
    :param targets: the ground truth outputs
    :return: a dict containing the mean squared error and mean absolute error
    """
    rMse = sqrt(((outputs - targets) ** 2).mean())
    mse = mean_squared_error(outputs, targets)
    mae = mean_absolute_error(outputs, targets)
    r2 = r2_score(targets, outputs)

    return {
        "rMse": rMse,
        "mse": mse,
        "mae": mae,
        "r2": r2
    }


if __name__ == "__main__":
    pass
