'''
Author: leixk_ai
Date: 2021-05-21 11:36:52
LastEditTime: 2021-05-21 13:50:19
LastEditors: Please set LastEditors
Description: 替换某个颜色像素值，可用来初步去除水印
FilePath: /img_utils/utils/modify_color.py
'''
import PIL
from PIL import Image
 
img = Image.open("E:\\fsdownload\\fsdownload\\bl_field_ann\\new\\000073.jpg")
print (img.size)
print(img.getpixel((4,4)))
 
width = img.size[0]#长度
height = img.size[1]#宽度
 
for w in range(0,width):
    for h in range(0,height):
        data = img.getpixel((w,h))#得到像素值
        if (163<data[0]<225 and 164<data[1]<225 and 159<data[2]<223):
        # if (data[0]==173 and data[1]==174 and data[2]==169):
            img.putpixel((w,h), (254  , 254  , 252))#则这些像素点的颜色改成大红色
 
img.save("y1.jpg")
img.show()