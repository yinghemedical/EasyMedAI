
import random
import numpy as np


def create_color_to_class(classList):
    color_to_class = {}
    for item in classList:
        r, g, b = random.randint(10, 240), random.randint(10, 240), random.randint(10, 240)
        color_to_class[(r, g, b)] = item
    return color_to_class

def convertNiiToPng(data,window_width,window_center):
    min = (2 * window_center - window_width) / 2.0 + 0.5
    max = (2 * window_center + window_width) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)
    data = data - min
    data =np.trunc( data * dFactor)
    data[data < 0.0] = 0
    data[data > 255.0] = 255 # 转换为窗位窗位之后的数据
    return data
    # # 将CT值转换为HU
    # hu_max = window_center + (window_width // 2)
    # hu_min = window_center - (window_width // 2)
    # data[data > hu_max] = hu_max
    # data[data < hu_min] = hu_min
    # data = (data - window_center) / (window_width / 2)
    # return data