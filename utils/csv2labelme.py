'''
Author: leixk_ai
Date: 2021-05-20 14:37:07
LastEditTime: 2021-06-03 18:45:30
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /img_utils/utils/labeme_json.py
'''

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import json
import numpy as np
import base64
import cv2
import json



def gen_polygon_json(img_path, out_path):
    """生成labelme能加载的json文件. 所有标注框都默认为polygon. 建议图像名称和json文件名称一致，只是后缀不一致.
    Args:
        boxes: [[[x1, y1], [x2, y2], ...], ...]，标注框.
        labels: [str1, str2, ...]，标注的labels.
        img_path: 图像路径.
        out_path: json文件输出路径.
    """
    # 确保标注框和标签能够一一对应
    # assert len(boxes) == len(labels), 'len(boxes) != len(labels)'

    EMPTY = {
        "version": "4.5.7",
        "flags": {},
        "shapes": [],
        "lineColor": [
            0,
            255,
            0,
            128
        ],
        "fillColor": [
            255,
            0,
            0,
            128
        ],
        "imagePath": "",
        "imageData": "",
        "imageHeight": 0,
        "imageWidth": 0
    }
    dict_ = EMPTY.copy()
    txt_file = img_path.split('.')[0] + ".txt"
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        member = line.strip('\n').split(',')
        x1, y1, x2, y2, x3, y3, x4, y4 = int(float(member[0])), int(float(member[1])), int(float(member[2])), int(float(member[3])),int(float(member[4])), int(float(member[5])), int(float(member[6])), int(float(member[7]))
        if (x1==x2 and y1 == y4 and y2 == y3 and x3==x4) or (x1==x4 and y1 == y2 and y3 == y4 and x2==x3) :
            shape = {
                "label": member[8],
                # "line_color": None,
                # "fill_color": None,
                # "points": [[float(pt[0]), float(pt[1])] for pt in xy],
                "points": [[x2, y2], [x4, y4]],
                # "group_id": "null",
                "shape_type": "rectangle"}
            # print('==>rectangle')
            dict_['shapes'].append(shape)
        else:
            shape = {
                "label": member[8],
                # "line_color": None,
                # "fill_color": None,
                # "points": [[float(pt[0]), float(pt[1])] for pt in xy],
                "points": [[x1, y1],[x2, y2],[x3, y3], [x4, y4]],
                # "group_id": "null",
                "shape_type": "polygon"}
            # print('==>polygon')
            dict_['shapes'].append(shape)

    # 记录图片路径和base64
    img_code = str(base64.b64encode(open(img_path, 'rb').read()), encoding='utf-8')
    img_name = os.path.split(img_path)[1]
    dict_['imagePath'] = img_name
    dict_['imageData'] = img_code

    # 记录图片size
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    dict_['imageHeight'] = h
    dict_['imageWidth'] = w
    # 输出json文件
    json.dump(dict_, open(out_path, 'w', encoding='utf-8'))


def main():
    i = 0
    path = r"E:\\fsdownload\\fsdownload\\bl_field_ann\\high"
    image_names = [file for file in os.listdir(path) if file.endswith('jpg')]
    for image_name in image_names:
        i += 1
        print(image_name)
        print(str(i))
        json_name = image_name[:image_name.rindex('.')] + '.json'
        gen_polygon_json(os.path.join(path, image_name), os.path.join(path, json_name))


main()


