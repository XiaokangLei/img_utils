'''
Author: leixk_ai
Date: 2021-05-19 14:25:58
LastEditTime: 2021-05-19 14:32:46
LastEditors: Please set LastEditors
Description: 将labelme生成的json文件转换成csv格式(x1, y1, x2, y2, x3, y3, x4, y4, label)的数据
FilePath: /img_utils/utils/labelme2csv.py
'''

import os
import json
import shutil
from argparse import ArgumentParser


def main():
    # 接收传参
    parser = ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='E:/fsdownload/fsdownload/bl_field_ann/val_org', help="labelme标注数据所在路径，图像和对应的json文件在同一路径，文件名前缀相同")
    parser.add_argument('--out_dir', type=str, default='E:/fsdownload/fsdownload/bl_field_ann/val', help="数据输出目录")
    parser.add_argument("--copy", type=int, default=1, help="是否复制图片到输出目录")  # 1代表复制
    args = parser.parse_args()

    # 创建文件输出路径
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    # 搜索输入路径下的json文件
    json_files = [name for name in os.listdir(args.in_dir) if name.endswith('.json')]

    # 遍历所有json文件
    for name in json_files:
        # 读取json文件信息
        print(name)
        try:
            d = json.loads(open(os.path.join(args.in_dir, name), 'r').read())
        except UnicodeDecodeError:
            d = json.loads(open(os.path.join(args.in_dir, name), 'r', encoding='gbk').read())
        stem = name[:-5]
        shapes = d['shapes']

        # 判断该图像是否没标，json文件没有shapes数据
        if len(shapes) == 0:
            print("%s shapes is null, skiped" % name)
            continue

        # 创建.txt文件，输出标注信息
        with open(os.path.join(args.out_dir, stem + '.txt'), 'w', encoding='utf-8') as f:
            for box in shapes:
                label = box['label']
                points = box['points']
                if len(points) == 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    if x1 < x2 and y1 < y2:
                        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    elif x1 > x2 and y1 < y2:
                        points = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
                    elif x1 > x2 and y1 > y2:
                        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    else:
                        points = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
                elif len(points) == 4:
                    pass
                else:
                    print('=> {}points数量异常.'.format(name))
                    continue
                loc = [a for pt in points for a in pt]
                loc = list(map(str, loc))
                loc.append(label)
                f.write(','.join(loc) + '\n')

        # 复制图像到输出路径
        if args.copy:
            shutil.copyfile(os.path.join(args.in_dir, d['imagePath']), os.path.join(args.out_dir, d['imagePath']))


if __name__ == '__main__':
    main()
