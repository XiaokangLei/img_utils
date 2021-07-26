'''
Author: leixk_ai
Date: 2021-06-17 09:00:41
LastEditTime: 2021-06-18 10:37:08
LastEditors: Please set LastEditors
Description: 条形码、二维码识别
FilePath: /img_utils/utils/img_pyzbar.py
'''
import pyzbar.pyzbar as pyzbar
from PIL import Image,ImageEnhance
import cv2
import numpy as np


image = "y1.jpg"

img = Image.open(image)

#img = ImageEnhance.Brightness(img).enhance(2.0)#增加亮度

#img = ImageEnhance.Sharpness(img).enhance(17.0)#锐利化

#img = ImageEnhance.Contrast(img).enhance(4.0)#增加对比度

#img = img.convert('L')#灰度化

img.show()

barcodes = pyzbar.decode(img)

for barcode in barcodes:
    barcodeData = barcode.data.decode("utf-8")
    (x, y, w, h) = barcode.rect#获取二维码的外接矩形顶点坐标
    print(barcodeData)
    # 二维码中心坐标
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    cv2.circle(img, (cx, cy), 2, (0, 255, 0), 8)  # 做出中心坐标
    print('中间点坐标：',cx,cy)

import cv2
import numpy as np
import copy
import math


def detecte(image):
    '''提取所有轮廓'''
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#// Convert Image captured from Image Input to GrayScale	
    canny = cv2.Canny(gray, 100, 200,3)#Apply Canny edge detection on the gray image
    contours,hierachy=cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#Find contours with hierarchy
    return contours,hierachy
    #coutour是一个list，每个元素都是一个轮廓（彻底围起来算一个轮廓），用numpy中的ndarray表示。
    #hierarchy也是一个ndarry，其中的元素个数和轮廓个数相同，每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，
    #分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数。
def conter(image):
    # 输入包含二维码的图片，返回二维码的实际位置以及朝向
    #图像的宽高和实际的距离
    image_width = 1920
    image_high = 1080
    map_width = 4.2
    map_high = 2.4
    contours,hierarchy=detecte(image)
    # print(len(contours))
    centers = {}
    M = {}
    # lista = [0,1,2,float("nan"),4]
    # print(lista[3])
    """
    Calculate the centers of the contours
    :param contours: Contours detected with find_contours
    :return: object centers as numpy array
    """
    for i in range(len(contours)):
        M[i] = cv2.moments(contours[i])
        if(M[i]["m00"] == 0):
            centers[i] = (float("nan"), float("nan"))
        else:
            centers[i] = (float(M[i]["m10"] / M[i]["m00"]), float(M[i]["m01"] / M[i]["m00"]))
        # print(centers[i])
  #计算符合5层嵌套的回形框
    mark = 0
    hierarchy = hierarchy[0]
    print(hierarchy[0])

    for i in range(len(contours)):
        k = i
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c = c+1
        if hierarchy[k][2] != -1:
            c = c+1
        if c >= 5:
            if mark == 0 :A = i
            elif mark == 1 :B = i
            elif mark == 2 :C = i
            mark = mark + 1
    # print(mark)
    #给返回值赋初值，如果没有识别到三个回形框，返回-1
    realPosition = [-1,-1]
    RotationAngle = -1
    if mark >=3:
        #计算三个回形框的质心点之间的距离
        AB = cv_distance(centers[A],centers[B])
        BC = cv_distance(centers[B],centers[C])
        CA = cv_distance(centers[C],centers[A])
        # print(AB)
        # print(BC)
        # print(CA)
        # print("three control points:",centers[A],centers[B],centers[C])
        #最长的斜边是right点和bottom点的连线，另一个点即为top点
        if AB > BC and AB > CA:
            outlier = C
            median1 = A
            median2 = B
        elif CA > AB and CA > BC:
            outlier = B
            median1 = A
            median2 = C
        elif BC > AB and BC > CA:
            outlier = A
            median1 = B
            median2 = C
        top = outlier
        # print("top point:",centers[top])
        
        #斜边的的中点就是二维码的质心
        CentralPoint_x = (centers[median1][0]+centers[median2][0])/2
        CentralPoint_y = (centers[median1][1]+centers[median2][1])/2
        CentralPoint = [CentralPoint_x,CentralPoint_y]
        # print("Central point:",CentralPoint)
        #根据图片像素与实际map的比例，可以求出二维码质心在实际中的位置
        #图片像素1920*1080  实际距离是4.2*2.4m
        realPosition_x =  (CentralPoint_x)/ image_width * map_width
        realPosition_y =  (CentralPoint_y) / image_high * map_high
        realPosition = [realPosition_x,realPosition_y]
        # print("real point:",realPosition)



			# //判断二维码旋转方向，通过求top点在对角线哪一侧
			# // 定义：平面上的三点A(x1,y1),B(x2,y2),C(x3,y3)的面积量：
			# // S(A,B,C)=|A B C|= (x1-x3)*(y2-y3)-(y1-y3)*(x2-x3)
			# // 令矢量的起点为A，终点为B，判断的点为C， 
			# // 如果S（A，B，C）为正数，则C在矢量AB的左侧； 
			# // 如果S（A，B，C）为负数，则C在矢量AB的右侧； 
			# // 如果S（A，B，C）为0，则C在直线AB上
        # 	//定义一个位于二维码左上方各200个像素的的DefaultTopPoint，#这样算出来的0度方向是水平向右
        # //当二维码旋转0度时，top点一定在此DefaultTopPoint与二维码质心CentralPoint的连线上
      
        DefaultTopPoint_x = CentralPoint_x + 200
        DefaultTopPoint_y = CentralPoint_y - 200
        DefaultTopPoint = [DefaultTopPoint_x,DefaultTopPoint_y]
        #沿逆时针方向增大到359
        Sdirection = (DefaultTopPoint_x - centers[top][0]) * (CentralPoint_y - centers[top][1]) -  (DefaultTopPoint_y - centers[top][1]) * (CentralPoint_x - centers[top][0])
        
        if Sdirection == 0:
            if centers[top][0]<CentralPoint_x: RotationAngle = 0#关键点的x坐标小于旋转中心的x坐标
            else:RotationAngle = 180
        else: 
            # //通过余弦定理，已知三边求角度
            aa = cv_distance(DefaultTopPoint,centers[top])
            bb = cv_distance(centers[top],CentralPoint)
            cc = cv_distance(CentralPoint,DefaultTopPoint)
            RotationAngle =  math.degrees(math.acos((aa*aa-bb*bb-cc*cc)/(-2*bb*cc)))#旋转角
            if Sdirection < 0: RotationAngle = 360-RotationAngle
        # print("RotationAngle:",RotationAngle)
    return realPosition,RotationAngle

#设计了一个函数调用接口，方便其他文件调用
#输入
# local_x,local_y剪裁出来的二维码部分的原点在原图中的位置(若未将原图进行剪裁则输入0，0即可)
# image 是输入的图片
# image_width,image_high 输入图片的宽高
# map_width,map_high  实际现实的距离
def Conter(local_x,local_y,image,image_width,image_high,map_width,map_high):
    # image_width = 1920
    # image_high = 1080
    # map_width = 4.2
    # map_high = 2.4
    contours,hierarchy=detecte(image)
    # print(len(contours))
    centers = {}
    M = {}
    # lista = [0,1,2,float("nan"),4]
    # print(lista[3])
    for i in range(len(contours)):
        M[i] = cv2.moments(contours[i])
        if(M[i]["m00"] == 0):
            centers[i] = (float("nan"), float("nan"))
        else:
            centers[i] = (float(M[i]["m10"] / M[i]["m00"]), float(M[i]["m01"] / M[i]["m00"]))
        # print(centers[i])

    mark = 0
    hierarchy = hierarchy[0]
    # print(hierarchy[0])

    for i in range(len(contours)):
        k = i
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c = c+1
        if hierarchy[k][2] != -1:
            c = c+1
        if c >= 5:
            if mark == 0 :A = i
            elif mark == 1 :B = i
            elif mark == 2 :C = i
            mark = mark + 1
    # print(mark)
    realPosition = [-1,-1]
    RotationAngle = -1
    if mark >=3:
        AB = cv_distance(centers[A],centers[B])
        BC = cv_distance(centers[B],centers[C])
        CA = cv_distance(centers[C],centers[A])
        # print(AB)
        # print(BC)
        # print(CA)
        # print("three control points:",centers[A],centers[B],centers[C])

        if AB > BC and AB > CA:
            outlier = C
            median1 = A
            median2 = B
        elif CA > AB and CA > BC:
            outlier = B
            median1 = A
            median2 = C
        elif BC > AB and BC > CA:
            outlier = A
            median1 = B
            median2 = C
        top = outlier
        # print("top point:",centers[top])
        

        CentralPoint_x = (centers[median1][0]+centers[median2][0])/2
        CentralPoint_y = (centers[median1][1]+centers[median2][1])/2
        CentralPoint = [CentralPoint_x,CentralPoint_y]
        # print("Central point:",CentralPoint)

        realPosition_x =  (CentralPoint_x+local_x)/ image_width * map_width
        realPosition_y =  (CentralPoint_y+local_y) / image_high * map_high
        realPosition = [realPosition_x,realPosition_y]
        # print("real point:",realPosition)

        # 	//定义一个位于二维码左上方各200个像素的的DefaultTopPoint，//
        #（+200，-200），默认朝右是零度。。（-200，-200）默认朝上是零度
        # //当二维码旋转0度时，top点一定在此DefaultTopPoint与二维码质心CentralPoint的连线上
        DefaultTopPoint_x = CentralPoint_x + 200
        DefaultTopPoint_y = CentralPoint_y - 200
        DefaultTopPoint = [DefaultTopPoint_x,DefaultTopPoint_y]

        Sdirection = (DefaultTopPoint_x - centers[top][0]) * (CentralPoint_y - centers[top][1]) -  (DefaultTopPoint_y - centers[top][1]) * (CentralPoint_x - centers[top][0])
        
        if Sdirection == 0:
            if centers[top][0]<CentralPoint_x: RotationAngle = 0#关键点的x坐标小于旋转中心的x坐标
            else:RotationAngle = 180
        else: 
            # //通过余弦定理，已知三边求角度
            aa = cv_distance(DefaultTopPoint,centers[top])
            bb = cv_distance(centers[top],CentralPoint)
            cc = cv_distance(CentralPoint,DefaultTopPoint)
            RotationAngle =  math.degrees(math.acos((aa*aa-bb*bb-cc*cc)/(-2*bb*cc)))#旋转角
            if Sdirection < 0: RotationAngle = 360-RotationAngle
        # print("RotationAngle:",RotationAngle)
    return realPosition,RotationAngle
def cv_distance(a,b):
    #求出两点的截距
   selfx=a[0]-b[0]
   selfy=a[1]-b[1]
   selflen= math.sqrt((selfx**2)+(selfy**2))
   return selflen




def run():
    print("start Reveive")
    # cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.115//stream1')
    # while True:
        # ret, frame = cap.read()
    frame=cv2.imread("3.jpg")
    realPosition,RotationAngle = conter(frame)
    print(realPosition,RotationAngle)
        # cv2.imshow('123456789',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

if __name__ == '__main__':

    run()
    pass