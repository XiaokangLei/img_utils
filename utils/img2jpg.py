'''
Author: leixk_ai
Date: 2021-05-21 11:01:28
LastEditTime: 2021-05-21 11:31:14
LastEditors: Please set LastEditors
Description: 图片批量转成jpg格式
FilePath: /img_utils/utils/img2jpg.py
'''

import cv2 as cv
import os
 
print('----------------------------------------------------')
print('程序的功能为：将该目录下的文件夹内的png格式的图片转为jpg')
print('转化为的结果: 在用户输入的文件夹名_1')
print('----------------------------------------------------')
print('')
 
# son = input('请输入该目录下文件夹名：')
# daddir= './'
# print(daddir)
# path = daddir + son+'\\'
path = 'E:\\fsdownload\\fsdownload\\bl_field_ann\\img_tmp\\'
 
# newpath = daddir+son+'_1'
newpath = 'E:\\fsdownload\\fsdownload\\bl_field_ann\\new\\'
if not os.path.exists(newpath):
    os.mkdir(newpath)
print(newpath)
 
path_list=os.listdir(path)
path_list.sort()
for filename in path_list:
    portion = os.path.splitext(filename)
    print('convert  ' + filename +'  to '+portion[0]+'.jpg')
    src = cv.imread(path+filename)
    cv.imwrite(newpath+'\\'+portion[0]+'.jpg',src)
 
print('转换完毕，文件存入 '+newpath+' 中')
cv.waitKey(0)
cv.destroyAllWindows()

# from PIL import Image
# from glob import glob
# import os

# def is_valid_image(img_path):
#     """
#     判断文件是否为有效（完整）的图片
#     :param img_path:图片路径
#     :return:True：有效 False：无效
#     """
#     valid = True
#     try:
#         Image.open(img_path).verify()
#     except:
#         valid = False
#     return valid


# def transimg(img_path):
#     """
#     转换图片格式
#     :param img_path:图片路径
#     :return: True：成功 False：失败
#     """
#     if is_valid_image(img_path):
#         try:
#             str_name = img_path.rsplit(".", 1)
#             output_img_path = str_name[0] + ".jpg"
#             print(output_img_path)
#             im = Image.open(img_path)
#             im.save(output_img_path)
#             if os.path.exists(img_path):
#                 os.remove(img_path)
#                 print('成功删除文件:', img_path)
#             else:
#                 print('未找到此文件:', img_path)
#             return True

#         except:
#             return False
#     else:
#         return False


# if __name__ == '__main__':
    # imgpath = glob('E:\\fsdownload\\fsdownload\\bl_field_ann\\img_tmp\\*.png')
    # for img in imgpath:
    #     _, imgfile = os.path.split(img)
    #     print(img,imgfile)
    #     img_path = img
    #     print(transimg(img_path))