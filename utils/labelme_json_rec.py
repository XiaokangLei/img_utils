'''
Author: leixk_ai
Date: 2021-05-20 15:01:29
LastEditTime: 2021-05-24 14:42:44
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /img_utils/utils/csv2labelme.py
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

from scipy.spatial import distance as dist
import numpy as np
import cv2
 
def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
 
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
 
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
 
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
 
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")

# 四边形点的顺时针排序
def order_points_new( pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	if leftMost[0, 1] != leftMost[1, 1]:
		leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	else:
		leftMost = leftMost[np.argsort(leftMost[:, 0])[::-1], :]
	(tl, bl) = leftMost
	if rightMost[0, 1] != rightMost[1, 1]:
		rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
	else:
		rightMost = rightMost[np.argsort(rightMost[:, 0])[::-1], :]
	(tr, br) = rightMost
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")
 

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
        # x1, y1, x2, y2, x3, y3, x4, y4 = float(member[0]), float(member[1]), float(member[2]), float(member[3]),float(member[4]), float(member[5]), float(member[6]), float(member[7])

        if (x1==x2 and y1 == y4 and y2 == y3 and x3==x4) or (x1==x4 and y1 == y2 and y3 == y4 and x2==x3) :
            x1, y1, x2, y2, x3, y3, x4, y4 = float(member[0]), float(member[1]), float(member[2]), float(member[3]),float(member[4]), float(member[5]), float(member[6]), float(member[7])
            tmp = order_points_new(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
            height = tmp[2][1] - tmp[0][1]
            tmp[0][0] = tmp[0][0] - 0.15* height
            tmp[0][1] = tmp[0][1] - 0.01* height
            tmp[2][0] = tmp[2][0] + 0.15* height
            tmp[2][1] = tmp[2][1] + 0.01* height
            shape = {
                "label": member[8],
                # "line_color": None,
                # "fill_color": None,
                # "points": [[float(pt[0]), float(pt[1])] for pt in xy],
                "points": [[np.float(tmp[0][0]),np.float(tmp[0][1])], [np.float(tmp[2][0]),np.float(tmp[2][1])]],
                # "group_id": "null",
                "shape_type": "rectangle"}
            # print('==>rectangle')
            dict_['shapes'].append(shape)
        else:
            x1, y1, x2, y2, x3, y3, x4, y4 = float(member[0]), float(member[1]), float(member[2]), float(member[3]),float(member[4]), float(member[5]), float(member[6]), float(member[7])

            tmp = order_points_new(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
            height = tmp[2][1] - tmp[0][1]
            tmp[0][0] = tmp[0][0] - 0.15* height
            tmp[0][1] = tmp[0][1] - 0.01* height
            tmp[2][0] = tmp[2][0] + 0.15* height
            tmp[2][1] = tmp[2][1] + 0.01* height
            tmp[1][0] = tmp[1][0] + 0.15* height
            tmp[1][1] = tmp[1][1] - 0.01* height
            tmp[3][0] = tmp[3][0] - 0.15* height
            tmp[3][1] = tmp[3][1] + 0.01* height
            shape = {
                "label": member[8],
                # "line_color": None,
                # "fill_color": None,
                # "points": [[float(pt[0]), float(pt[1])] for pt in xy],
                "points": [[np.float(tmp[0][0]),np.float(tmp[0][1])], [np.float(tmp[1][0]),np.float(tmp[1][1])],[np.float(tmp[2][0]),np.float(tmp[2][1])], [np.float(tmp[3][0]),np.float(tmp[3][1])]],
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
    path = r"E:\fsdownload\fsdownload\bl_field_ann\val"
    image_names = [file for file in os.listdir(path) if file.endswith('jpg')]
    for image_name in image_names:
        i += 1
        print(image_name)
        print(str(i))
        json_name = image_name[:image_name.rindex('.')] + '.json'
        gen_polygon_json(os.path.join(path, image_name), os.path.join(path, json_name))


main()


