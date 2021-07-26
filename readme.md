# 记录python实现图像处理

## 文件目录及功能

- img 待测试图片目录
- pic readme图片文件
- utils 图像处理功能文件
  - csv2labelme.py txt转labelme可以加载的json文件
  - draw_point.py 图像上画点
  - img_enhance.py 图像增强
  - img_pyzbar.py 条形码、二维码识别
  - img2jpg.py 图片批量转成jpg格式
  - labelme_json_rec.py
  - labelme2csv.py 将labelme生成的json文件转换成csv格式(x1, y1, x2, y2, x3, y3, x4, y4, label)的数据
  - modify_color.py 替换某个颜色像素值，可用来初步去除水印
  - rename_img.py 批量重命名图片
  - text_rotate.py 根据文本方向调整图片
