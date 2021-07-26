'''
Author: your name
Date: 2021-06-09 13:46:16
LastEditTime: 2021-06-09 18:16:47
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /img_utils/utils/draw_point.py
'''

from PIL import Image
from pylab import imshow
from pylab import array
from pylab import plot
from pylab import title
 
#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@Time    : 2018-11-09 21：39
@Author  : jianjun.wang
@Email   : alanwang6584@gmail.com
"""

import numpy as np
import cv2 as cv
import math
 

# 旋转矩阵转欧拉角
def isRotationMatrix(R):
    Rt = np.transpose(R)   #旋转矩阵R的转置
    shouldBeIdentity = np.dot(Rt, R)   #R的转置矩阵乘以R
    I = np.identity(3, dtype=R.dtype)           # 3阶单位矩阵
    n = np.linalg.norm(I - shouldBeIdentity)   #np.linalg.norm默认求二范数
    return n < 1e-6                            # 目的是判断矩阵R是否正交矩阵（旋转矩阵按道理须为正交矩阵，如此其返回值理论为0）
 
 
def rotationMatrixToAngles(R):
    assert (isRotationMatrix(R))   #判断是否是旋转矩阵（用到正交矩阵特性）
 
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])  #矩阵元素下标都从0开始（对应公式中是sqrt(r11*r11+r21*r21)），sy=sqrt(cosβ*cosβ)
 
    singular = sy < 1e-6   # 判断β是否为正负90°
 
    if not singular:   #β不是正负90°
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:              #β是正负90°
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)   #当z=0时，此公式也OK，上面图片中的公式也是OK的
        z = 0
    
    x = x*180.0/3.141592653589793
    y = y*180.0/3.141592653589793
    z = z*180.0/3.141592653589793
 
    return np.array([x, y, z])

img = np.zeros((640, 640, 3), np.uint8)+255 #生成一个空灰度图像
print(img.shape) # 输出：(480, 480, 3)

point_size = 1
point_color = (0, 0, 255) # BGR
thickness = 2 # 可以为 0 、4、8

# 要画的点的坐标
points_list = [(271,529),(163,273),(293,432),(180,288),(307,443),(112,396),(107,375),(103,353),(97,331),(94,309),(96,287),(98,264),(162,287),(384,247),(389,289),(387,268),(389,330),(390,309),(384,371),(121,241),(352,267),(253,268),(257,323),(255,296),(280,228),(293,428),(278,410),(293,275),(224,230),(170,219),(289,452),(218,368),(266,457),(302,418),(179,268),(230,281),(264,430),(225,423),(263,415),(248,412),(236,432),(321,259),(313,430),(323,278),(283,240),(205,437),(260,381),(340,274),(352,225),(378,412),(322,430),(307,277),(321,267),(197,271),(306,232),(179,277),(382,391),(181,506),(196,239),(211,281),(222,244),(214,437),(145,237),(235,435),(242,455),(222,448),(126,439),(196,285),(387,350),(260,352),(327,211),(329,225),(179,276),(302,215),(373,224),(353,211),(245,529),(222,525),(200,517),(199,222),(163,492),(148,476),(135,459),(170,235),(142,223),(338,261),(289,372),(275,376),(231,376),(245,378),(226,342),(275,277),(289,337),(300,362),(149,281),(314,515),(330,502),(294,524),(365,452),(374,433),(344,487),(357,471),(118,418),(264,434),(304,265),(321,267)]
print("==> ",points_list[7][0],points_list[69][0],points_list[16][0])
# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                         
                        ])
#2D image points. If you change the image, you need to change vector
image_points = np.array([
                            points_list[0],     # Nose tip  				NO.1
                            points_list[69],     # Chin 					NO.70
                            points_list[94],     # Left eye left corner 	NO.95
                            points_list[20],     # Right eye right corne 	NO.21
                            points_list[45],     # Left Mouth corner 		NO.46
                            points_list[50]      # Right mouth corner 		NO.51
                        ], dtype="double")

# Camera internals
 
focal_length = img.shape[1]
center = (img.shape[1]/2, img.shape[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
 
print("Camera Matrix :\n {0}".format(camera_matrix))
 
dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
(success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
 
print("Rotation Vector:\n {0}".format(rotation_vector))
print("Translation Vector:\n {0}".format(translation_vector))

# 旋转向量转旋转矩阵
theta = np.linalg.norm(rotation_vector)
r = rotation_vector / theta
R_ = np.array([[0, -r[2][0], r[1][0]],
               [r[2][0], 0, -r[0][0]],
               [-r[1][0], r[0][0], 0]])
R = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * r * r.T + np.sin(theta) * R_
print('旋转矩阵')
print(R)

result = rotationMatrixToAngles(R)
print("==> result:",result)

index = 1
for point in points_list:
	cv.circle(img, point, point_size, point_color, thickness)
	cv.putText(img, str(index), point, cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv.LINE_AA, bottomLeftOrigin=None)
	index += 1

# 画圆，圆心为：(160, 160)，半径为：60，颜色为：point_color，实心线
# cv.circle(img, (160, 160), 60, point_color, 0)
cv.imwrite('./rest.jpg',img)
cv.namedWindow("image")
cv.imshow('image', img)
cv.waitKey (5000) # 显示 10000 ms 即 10s 后消失
cv.destroyAllWindows()
