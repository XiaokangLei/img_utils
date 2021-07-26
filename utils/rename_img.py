'''
Author: leixk_ai
Date: 2021-05-21 11:13:27
LastEditTime: 2021-06-04 17:47:08
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /img_utils/utils/rename_img.py
'''

# -*- coding:utf8 -*-

import os
from PIL import Image

class BatchRename():
    '''
    批量重命名文件夹中的图片文件

    '''
    def __init__(self):
        self.path = r'F:\\train_chengdu\\code\\gitlab\\BLOCR-master\\data\\test'  #表示需要命名处理的文件夹

    def is_valid_image(self,img_path):
        """
        判断文件是否为有效（完整）的图片
        :param img_path:图片路径
        :return:True：有效 False：无效
        """
        bvalid = True
        try:
            Image.open(img_path).verify()
        except:
            bvalid = False
        return bvalid


    def transimg(self,img_path):
        """
        转换图片格式
        :param img_path:图片路径
        :return: True：成功 False：失败
        """
        if self.is_valid_image(img_path):
            try:
                str_name = img_path.rsplit(".", 1)
                output_img_path = str_name[0] + ".jpg"
                print(output_img_path)
                im = Image.open(img_path)
                im.save(output_img_path)
                return True
            except:
                return False
        else:
            return False
            

    def rename(self):
        #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序
        filelist = os.listdir(self.path)
        total_num = len(filelist) #获取文件夹内所有文件个数
        i = 0  #表示文件的命名是从1开始的
        for item in filelist:
            if item.endswith('.jpg') or item.endswith('.jpeg'):
                print("item_jpg:",item)
                # with open('./id_test/label.txt', 'r+') as f:
                #     content = f.read()        
                #     f.seek(0, 0)
                #     f.write(item + ',1\n' +content)
                # #初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的即可）
                src = os.path.join(os.path.abspath(self.path), item)
                # dst = os.path.join(os.path.abspath(self.path),str(i) + '.jpg')
                #处理后的格式也为jpg格式的，当然这里可以改成png格式
                dst = os.path.join(os.path.abspath(self.path), '00' + format(str(i), '0>4s') + '.jpg')#这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式
                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    i = i + 1
                    continue
            elif item.endswith('.png'):
                print("item_png:",item)
                # with open('./id_test/label.txt', 'r+') as f:
                #     content = f.read()        
                #     f.seek(0, 0)
                #     f.write(item + ',1\n' +content)
                # #初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的即可）
                src = os.path.join(os.path.abspath(self.path), item)
                # dst = os.path.join(os.path.abspath(self.path),str(i) + '_recapture.jpg')
                #处理后的格式也为jpg格式的，当然这里可以改成png格式
                dst = os.path.join(os.path.abspath(self.path), '00' + format(str(i), '0>4s') + '.png')#这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式
                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    i = i + 1
                    continue
        print ('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()