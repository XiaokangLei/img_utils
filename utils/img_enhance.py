'''
Author: leixk_ai
Date: 2021-05-28 10:57:45
LastEditTime: 2021-05-28 14:20:39
LastEditors: Please set LastEditors
Description: 图像增强
FilePath: /img_utils/utils/img_enhance.py
'''
# coding: utf-8
import numpy as np
import cv2
import random
import math 

# 运动模糊，可随机设置degree 4-6,angle -45-45
def motion_blur(image):
    image = np.array(image)
    degree = random.randint(4,6)
    angle = random.randint(-45,45)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred
# img = cv2.imread('./y1.jpg')
# img_ = motion_blur(img)
# cv2.imshow('Source image',img)
# cv2.imshow('blur image',img_)
# cv2.waitKey()

# 高斯模糊
# img = cv2.imread('./y1.jpg')
# img_ = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0, sigmaY=0)
# cv2.imshow('Source image',img)
# cv2.imshow('blur image',img_)
# cv2.waitKey()
# x, y = img.shape[0:2]   # 获取图像的宽和高
# img_test1 = cv2.resize(img, (int(y), int(x)),interpolation=cv2.INTER_NEAREST)  # 注意x，y的顺序不要写满
# cv2.imwrite('l.jpg', img_test1)
# cv2.imshow('Source image',img)
# cv2.imshow('blur image',img_test1)
# cv2.waitKey()

def rotate_image(image, degree):
    # 逆时针旋转图像degree角度（原尺寸）
    # 旋转中心为图像中心
    h, w = image.shape[:2]
    # 计算二维旋转的仿射变换矩阵
    RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)
    # print(RotateMatrix)
    # 仿射变换，背景色填充为白色
    rotate = cv2.warpAffine(image, RotateMatrix, (w, h), borderValue=(255, 255, 255))
    return rotate, RotateMatrix
def remote(img, degree):
    height, width = img.shape[:2]
    radians = float(degree / 180 * np.pi)
    heightNew = int(width * math.fabs(math.sin((radians))) + height * math.fabs(math.cos((radians))))
    widthNew = int(height * math.fabs(math.sin((radians))) + width * math.fabs(math.cos((radians))))
    # 得到二维矩阵的旋转的仿射矩阵
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    # 中心位置的实际平移
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation,matRotation
img = cv2.imread('./y1.jpg')
img_copy = cv2.imread('./y1.jpg')
x=202
y=545
xx=562
yy=782
img_new,matrix=rotate_image(img,15)
cv2.rectangle(img,(x,y),(xx,yy),(255,0,0),3,4,0)#用rectangle对图像进行画框
img_new1,matrix1=remote(img_copy,-6)
result = np.dot(matrix1, [[x],[y],[1]]).astype(np.int)
result1 = np.dot(matrix1, [[xx],[yy],[1]]).astype(np.int)
print("==>result:",result)
p_re=[]
x=result[0][0]
y=result[1][0]
xx=result1[0][0]
yy=result1[1][0]
p_re.append([x,y])
p_re.append([xx,yy])
cv2.rectangle(img_new1,(x,y),(xx,yy),(255,0,0),3,4,0)#用rectangle对图像进行画框

print("==>p_re:",p_re)
cv2.imshow('Source image',img)
cv2.imshow('blur image',img_new)
cv2.imshow('blur image2',img_new1)
cv2.waitKey()