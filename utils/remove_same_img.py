'''
Author: leixk_ai
Date: 2021-05-26 11:36:10
LastEditTime: 2021-05-28 09:04:27
LastEditors: Please set LastEditors
Description: 去除相同的图片
FilePath: /img_utils/utils/remove_same_img.py
'''
#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
from PIL import Image
from argparse import ArgumentParser


def list_images(img_dir, postfixes):
    image_names = []
    for file in os.listdir(img_dir):
        for pf in postfixes:
            if file.endswith(pf):
                image_names.append(file)
                break
    return image_names


def find():
    """查找重复的图片."""
    postfixes = args.postfixes.split(',')

    img_names = list_images(args.img_dir, postfixes)

    img_dict = {}

    for file in img_names:
        img_path = os.path.join(args.img_dir, file)
        file_size = os.path.getsize(img_path)
        try:
            img = Image.open(img_path)
        except OSError as e:
            print('OSError:', e)
            continue
        w, h = img.size
        key = '{},{},{}'.format(file_size, w, h)
        if key in img_dict:
            img_dict[key].append(file)
        else:
            img_dict[key] = [file]

    print("=> 重复的图像：")
    for k, v in img_dict.items():
        if len(v) > 1:
            print(v)


def find_and_remove_repeat():
    """查找并删除重复的图片."""
    postfixes = args.postfixes.split(',')

    img_names = list_images(args.img_dir, postfixes)

    img_dict = {}

    for file in img_names:
        img_path = os.path.join(args.img_dir, file)
        file_size = os.path.getsize(img_path)
        try:
            img = Image.open(img_path)
        except OSError as e:
            print(file)
            print('OSError:', e)
            continue
        w, h = img.size
        key = '{},{},{}'.format(file_size, w, h)
        if key in img_dict:
            img_dict[key].append(file)
        else:
            img_dict[key] = [file]

    print("=> 重复的图像：")
    for k, files in img_dict.items():
        if len(files) > 1:
            print("=> keep: \n" + files[0])
            print("=> remove: ")
            for file in files[1:]:
                print(os.path.join(args.img_dir, file))
                os.remove(os.path.join(args.img_dir, file))


if __name__ == '__main__':
    parser = ArgumentParser('find_repeated_images')
    parser.add_argument('--img_dir', type=str, default='E:/fsdownload/fsdownload/bl_field_ann/business_license/business_license/horizontal',help="图片路径.")
    parser.add_argument('--postfixes', '-pf', type=str, default='.jpg,.png,.jpeg,.bmp', help="图片后缀.")

    args = parser.parse_args()

    find()
    # find_and_remove_repeat()
