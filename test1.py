'''
Author: your name
Date: 2021-05-20 15:27:03
LastEditTime: 2021-05-24 11:29:20
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /img_utils/test1.py
'''
# import the necessary packages
from scipy.spatial import distance as dist
import numpy as np
import cv2

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
# [11,23,24,23,24,36,11,36]
print(order_points_new(np.array([
        [
          302.9585798816568,
          1397.6331360946747
        ],
        [
          918.9349112426036,
          1390.5325443786983
        ],
        [
          918.3431952662722,
          1413.6094674556214
        ],
        [
          301.7751479289941,
          1421.8934911242604
        ]
      ])))