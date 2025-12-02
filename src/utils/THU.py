#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/24 22:22
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   THU.py
# @Desc     :   

from thulac import thulac

from src.utils.decorator import timer

# Initialize THULAC with segmentation only or not
thu_false = thulac(seg_only=False)
thu_true = thulac(seg_only=True)


@timer
def cut_pos(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """ Cut text using THULAC
    :param text: text to cut
    :return: list of tuples of cut words and their POS tags
    """
    words_tag: list[tuple[str, str]] = thu_false.cut(text)
    words: list[str] = [word for word, tag in words_tag]

    print(f"The text has been cut into {len(words)} words using THULAC.")

    return words_tag, words


# @timer
def cut_only(text: str) -> list[str]:
    """ Cut text using THULAC without POS tags
    :param text: text to cut
    :return: list of cut words
    """
    words: list[str] = thu_true.cut(text)
    # Flatten the list of tuples to get only words
    words = [word for word, _ in words]

    # print(f"The text has been cut into {len(words)} words using THULAC (without POS tags).")

    return words


if __name__ == "__main__":
    pass
