# coding:utf-8
import re


def isCapital(word):
    if re.match('^[A-Z]+', word):
        return 2
    elif re.match('^[a-zA-Z]+', word):
        return 3
    elif re.match('^[0-9a-zA-Z]+',word):
        return 4
    else:
        return 5

