'''
Author: leixk_ai
Date: 2021-05-31 09:56:28
LastEditTime: 2021-05-31 11:27:19
LastEditors: Please set LastEditors
Description: 根据文本方向调整
FilePath: /img_utils/utils/text_rotate.py
'''

# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt
# import cv2

# # 度数转换
# def degree_trans(theta):
#     res = theta / np.pi * 180
#     return res

# # 逆时针旋转图像degree角度（原尺寸）
# def rotate_image(src, degree):
#     # 旋转中心为图像中心
#     h, w = src.shape[:2]
#     # 计算二维旋转的仿射变换矩阵
#     rotate_matrix = cv2.getRotationMatrix2D((w/2.0, h/2.0), degree, 1)
#     print(rotate_matrix)
#     # 仿射变换，背景色填充为白色
#     rotate = cv2.warpAffine(src, rotate_matrix, (w, h), borderValue=(255, 255, 255))
#     return rotate

# # 通过霍夫变换计算角度
# def calc_degree(src_mage):
#     mid_image = cv2.cvtColor(src_mage, cv2.COLOR_BGR2GRAY)
#     dst_image = cv2.Canny(mid_image, 50, 200, 3)
#     lineimage = src_mage.copy()
 
#     # 通过霍夫变换检测直线
#     # 第4个参数就是阈值，阈值越大，检测精度越高
#     lines = cv2.HoughLines(dst_image, 1, np.pi/180, 200)
#     # 由于图像不同，阈值不好设定，因为阈值设定过高导致无法检测直线，阈值过低直线太多，速度很慢
#     sumth = 0
#     # 依次画出每条线段
#     plt.figure()
#     for i in range(len(lines)):
#         for rho, theta in lines[i]:
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             x1 = int(round(x0 + 1000 * (-b)))
#             y1 = int(round(y0 + 1000 * a))
#             x2 = int(round(x0 - 1000 * (-b)))
#             y2 = int(round(y0 - 1000 * a))
#             # 只选角度最小的作为旋转角度
#             sumth += theta
#             cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
#             plt.subplot(222),plt.imshow(lineimage)
#             cv2.imshow("image", lineimage)
 
#     # 对所有角度求平均，这样做旋转效果会更好
#     average = sumth / len(lines)
#     angle = degree_trans(average) - 90
#     return angle

# input_img_file = "./23.jpg"
# image = cv2.imread(input_img_file)
# w,h,_ = image.shape
# print("==>image.shape:",image.shape)
# plt.subplot(221),plt.imshow(image)
# cv2.imshow("src image", image)
# #[h//3:2*h//3,w//3:2*h//3]
# degree = calc_degree(image)
# print("调整角度：", degree)
# rotate = rotate_image(image, degree)
# cv2.imshow("dest image", rotate)
# plt.subplot(223),plt.imshow(rotate)
# plt.show
# cv2.waitKey(0)


# -*- coding: UTF-8 -*-
 
# import numpy as np
# import cv2
 
# ## 图片旋转
# def rotate_bound(image, angle):
#     #获取宽高
#     (h, w) = image.shape[:2]
#     (cX, cY) = (w // 2, h // 2)
 
#     # 提取旋转矩阵 sin cos 
#     M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])
 
#     # 计算图像的新边界尺寸
#     nW = int((h * sin) + (w * cos))
# #     nH = int((h * cos) + (w * sin))
#     nH = h
 
#     # 调整旋转矩阵
#     M[0, 2] += (nW / 2) - cX
#     M[1, 2] += (nH / 2) - cY
 
#     return cv2.warpAffine(image, M, (nW, nH),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
 
# ## 获取图片旋转角度
# def get_minAreaRect(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.bitwise_not(gray)
#     thresh = cv2.threshold(gray, 0, 255,
#         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#     coords = np.column_stack(np.where(thresh > 0))
#     return cv2.minAreaRect(coords)
 
# image_path = "./222.jpg"
# image = cv2.imread(image_path)
# angle = get_minAreaRect(image)[-1]
# rotated = rotate_bound(image, angle)
 
# cv2.putText(rotated, "angle: {:.2f} ".format(angle),
#     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
# # show the output image
# print("[INFO] angle: {:.3f}".format(angle))
# cv2.imshow("imput", image)
# cv2.imshow("output", rotated)
# cv2.waitKey(0)


# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# def rotate_bound(image, angle):# 获取图片的宽高
#     (h, w) = image.shape[:2]
#     (cX, cY) = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
#     img = cv2.warpAffine(image, M, (w, h))
#     return img

# def rotate_points(points, angle, cX, cY):
#     M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0).astype(np.float16)
#     a = M[:, :2]
#     b = M[:, 2:]
#     b = np.reshape(b, newshape=(1, 2))
#     a = np.transpose(a)
#     points = np.dot(points, a) + b
#     points = points.astype(np.int)
#     return points

# def findangle(_image):
#     toWidth = _image.shape[1]//2 #500
#     minCenterDistance = toWidth/20 #10
#     angleThres = 45

#     image = _image.copy()
#     h, w = image.shape[0:2]
#     if w > h:
#         maskW = toWidth
#         maskH = int(toWidth / w * h)
#     else:
#         maskH = toWidth
#         maskW = int(toWidth / h * w)
#     # 使用黑色填充图片区域
#     swapImage = cv2.resize(image, (maskW, maskH))
#     grayImage = cv2.cvtColor(swapImage, cv2.COLOR_BGR2GRAY)
#     gaussianBlurImage = cv2.GaussianBlur(grayImage, (3, 3), 0, 0)
#     histImage = cv2.equalizeHist(~gaussianBlurImage)
#     binaryImage = cv2.adaptiveThreshold(histImage, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
    
#     pointsNum = np.sum(binaryImage!=0)//2

#     connectivity = 8
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaryImage, connectivity, cv2.CV_8U)
#     labels = np.array(labels)
#     maxnum = [(i, stats[i][-1], centroids[i]) for i in range(len(stats))]
#     maxnum = sorted(maxnum, key=lambda s: s[1], reverse=True)
#     if len(maxnum) <= 1:
#         return 0
#     for i, (label, count, centroid) in enumerate(maxnum[1:]):
#         cood = np.array(np.where(labels == label))
#         distance1 = np.linalg.norm(cood[:,0]-centroid[::-1])
#         distance2 = np.linalg.norm(cood[:,-1]-centroid[::-1])
#         if distance1 > minCenterDistance or distance2 > minCenterDistance:
#             binaryImage[labels == label] = 0
#         else:
#             break

#     minRotate = 0
#     minCount = -1
#     (cX, cY) = (maskW // 2, maskH // 2)
#     points = np.column_stack(np.where(binaryImage > 0))[:pointsNum].astype(np.int16)
#     for rotate in range(-angleThres, angleThres):
#         rotatePoints = rotate_points(points, rotate, cX, cY)
#         rotatePoints = np.clip(rotatePoints[:,0], 0, maskH-1)
#         hist, bins = np.histogram(rotatePoints, maskH, [0, maskH])
#         # 横向统计非零元素个数 越少则说明姿态越正
#         zeroCount = np.sum(hist > toWidth/50)
#         if zeroCount <= minCount or minCount == -1:
#             minCount = zeroCount
#             minRotate = rotate
#     return minRotate

# Path = '222.jpg'
# cv_img = cv2.imdecode(np.fromfile(Path, dtype=np.uint8), -1)
# cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

# for agl in range(-30, 30):
#     img = cv_img.copy()
#     img = rotate_bound(img, agl)
#     angle = findangle(img)
#     img = rotate_bound(img, -angle)
#     angle = findangle(img)
#     cv2.imshow('after', img)
#     key = cv2.waitKey(1) & 0xFF
#     # 按'q'健退出循环
#     if key == ord('q'):
#         break
# cv2.destroyAllWindows()
        
# plt.imshow(img)
# plt.show()

# import cv2 as cv
# import numpy as np
# import math
# from matplotlib import pyplot as plt

# def fourier_demo():
#     #1、读取文件，灰度化
#     img = cv.imread('./22.jpg')
#     cv.imshow('original', img)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     cv.imshow('gray', gray)

#     #2、图像延扩
#     h, w = img.shape[:2]
#     new_h = cv.getOptimalDFTSize(h)
#     new_w = cv.getOptimalDFTSize(w)
#     right = new_w - w
#     bottom = new_h - h
#     nimg = cv.copyMakeBorder(gray, 0, bottom, 0, right, borderType=cv.BORDER_CONSTANT, value=0)
#     cv.imshow('new image', nimg)

#     #3、执行傅里叶变换，并过得频域图像
#     f = np.fft.fft2(nimg)
#     fshift = np.fft.fftshift(f)
#     magnitude = np.log(np.abs(fshift))


#     #二值化
#     magnitude_uint = magnitude.astype(np.uint8)
#     ret, thresh = cv.threshold(magnitude_uint, 11, 255, cv.THRESH_BINARY)
#     print(ret)
#     cv.imshow('thresh', thresh)
#     print(thresh.dtype)
#     #霍夫直线变换
#     lines = cv.HoughLinesP(thresh, 2, np.pi/180, 30, minLineLength=40, maxLineGap=100)
#     print(len(lines))

#     #创建一个新图像，标注直线
#     lineimg = np.ones(nimg.shape,dtype=np.uint8)
#     lineimg = lineimg * 255

#     piThresh = np.pi/180
#     pi2 = np.pi/2
#     print(piThresh)

#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv.line(lineimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         if x2 - x1 == 0:
#             continue
#         else:
#             theta = (y2 - y1) / (x2 - x1)
#         if abs(theta) < piThresh or abs(theta - pi2) < piThresh:
#             continue
#         else:
#             print(theta)

#     angle = math.atan(theta)
#     print(angle)
#     angle = angle * (180 / np.pi)
#     print(angle)
#     angle = (angle - 90)/(w/h)
#     print(angle)

#     center = (w//2, h//2)
#     M = cv.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
#     cv.imshow('line image', lineimg)
#     cv.imshow('rotated', rotated)

# fourier_demo()
# cv.waitKey(0)
# cv.destroyAllWindows()
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from pandas import Series, DataFrame
# img = cv2.imread('./22.jpg')
# plt.imshow(img)
# plt.show()
# grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret,binImg = cv2.threshold(grayImg, 100, 255, cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(binImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0, 200, 255), 2)
# plt.imshow(img)
# plt.show()
# hierarchyDF = DataFrame(hierarchy[0], columns = ['pre', 'next', 'child', 'parent'])

#! /usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
 
 
# 完成灰度化，二值化
def two_value(img_raw):
    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)      # 灰度化
    ret, img_two = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)     # 二值化
    return img_two
 
 
# 旋转函数
def rotate(img_rotate_raw, angle):
    (h, w) = img_rotate_raw.shape[:2]
    (cx, cy) = (w//2, h//2)
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)  # 计算二维旋转的仿射变换矩阵
    return cv2.warpAffine(img_rotate_raw, m, (w, h), borderValue=(0, 0, 0))
 
 
# 霍夫直线检测
def get_angle(img_hou_raw):
    sum_theta = 0
    img_canny = cv2.Canny(img_hou_raw, 50, 200, 3)
    lines = cv2.HoughLines(img_canny, 1, np.pi/180, 300, 0, 0)
    # lines 是三维的
    for i in range(lines.shape[0]):
        theta = lines[i][0][1]
        sum_theta += theta
    average = sum_theta / lines.shape[0]
    angle = average/np.pi*180 - 90
    return angle
 
 
def correct(img_cor_raw):
    img_two = two_value(img_cor_raw)
    angle = get_angle(img_two)
    if angle == -1:
        print("No lines!!!")
        return 0
    return rotate(img_two, angle)
 
 
if __name__ == "__main__":
    img = cv2.imread("./22.jpg")
    cv2.imshow("raw", img)
    img_rot2 = correct(img)
    cv2.imshow("last", img_rot2)
    cv2.waitKey()