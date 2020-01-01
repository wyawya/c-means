#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import cv2

# 读入黑背景下的彩色手写数字
img = cv2.imread("/home/heyue/fcm_segmentation/fcm_result.jpg")
img1 = cv2.imread("/home/heyue/IMG_20190715_172751.jpg")
#img1 = cv2.resize(img1,(100,75))
# 转换为gray灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
binary=255-binary
cv2.imwrite("s.png", binary)
cv2.imshow('img',img)
cv2.waitKey(0)
# 寻找轮廓
contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 声明画布 拷贝自img
canvas = np.copy(img1)
#canvas = np.copy(binary)

x_lefttop=75
y_lefttop=100
x_rightbottom=0
y_rightbottom=0

for cidx,cnt in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(cnt)
   # print('RECT: x={}, y={}, w={}, h={}'.format(x, y, w, h))
    #去除小连通区域
    if w*h<50:
        continue

    if x<x_lefttop:
        x_lefttop=x
    if y<y_lefttop:
        y_lefttop=y
    if x_rightbottom<x+w:
        x_rightbottom=x+w
    if y_rightbottom<y+h:
        y_rightbottom=y+h
    # 原图绘制圆形
    cv2.rectangle(canvas, pt1=(x*10, y*10), pt2=(x*10 + w*10, y*10 + h*10), color=(100, 100, 255), thickness=3)
    # 截取ROI图像
    #cv2.imwrite("number_boudingrect_cidx_{}.png".format(cidx), img[y:y+h, x:x+w])

x_lefttop=x_lefttop*10
y_lefttop=y_lefttop*10
x_rightbottom=x_rightbottom*10
y_rightbottom=y_rightbottom*10
#cv2.rectangle(canvas, pt1=(x_lefttop, y_lefttop), pt2=(x_rightbottom, y_rightbottom), color=(100, 100, 255), thickness=3)

cv2.imwrite("number_boundingrect_canvas.png", canvas)