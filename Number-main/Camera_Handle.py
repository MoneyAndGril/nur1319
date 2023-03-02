import numpy as np
from PIL import Image
import cv2

# maxValue = cv2.imread('./pic\\TelCamera\\22.jpg')
# cv2.imshow('img', img)
# cv2.waitKey(0)

# maxValue = img.copy()

# img_gray = cv2.cvtColor(maxValue, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img',img)
# ret, imgviewx2 =cv2.threshold(img1, 70, 255,cv2.THRESH_TOZERO)

'''
    cv2.THRESH_BINARY
    cv2.THRESH_OTSU
'''

'''
    cv2.adaptiveThreshold(img, maxValue, adaptiveMethod, thresholdType, blockSize, C)
'''

# output1 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
#
#
# output2 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# cv2.resize()
# print('gray_shape:'+str(img_gray.shape))
# print('gray_MinMax:'+str(output1.shape))

# cv2.imshow('img1', output1)
# cv2.waitKey(0)
# cv2.imshow('img2', output2)
# cv2.waitKey(0)

'''图片模糊化'''
# img_gray2 = cv2.medianBlur(img_gray, 5)
# cv2.imshow('img', img_gray2)
# cv2.waitKey(0)
# output2 = cv2.adaptiveThreshold(img_gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
# cv2.imshow('img1', output2)
# cv2.waitKey(0)

# gray=cv2.cvtColor(maxValue,cv2.COLOR_BGR2GRAY)
# element=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# # gray = output2
# # 二次腐蚀处理
# gray2=cv2.dilate(gray,element)
# gray3=cv2.dilate(gray2,element)
# cv2.imshow("dilate", gray3)
# cv2.waitKey(0)
# # 二次膨胀处理
# gray2=cv2.erode(gray,element)
# gray2=cv2.erode(gray2,element)
# cv2.imshow("erode", gray2)
# cv2.waitKey(0)
# # 膨胀腐蚀做差
# edges=cv2.absdiff(gray,gray2)
# cv2.imshow("absdiff", edges)
# cv2.waitKey(0)

# cv2.imwrite('./pic\\TelCamera\\gao.jpg',edges)

def where_num(frame):
    rois=[]

    # 灰度处理
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    element=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    # 二次腐蚀处理
    gray2=cv2.dilate(gray,element)
    gray3=cv2.dilate(gray2,element)

    # 二次膨胀处理
    gray2 = cv2.erode(gray, element)
    gray4=cv2.erode(gray2,element)    #膨胀的gray2

    # 膨胀腐蚀做差
    edges=cv2.absdiff(gray4,gray3)

    # 使用算子进行降噪
    x=cv2.Sobel(edges,cv2.CV_16S,1,0)
    y=cv2.Sobel(edges,cv2.CV_16S,0,1)
    absX=cv2.convertScaleAbs(x)
    absY=cv2.convertScaleAbs(y)
    dst=cv2.addWeighted(absX,0.5,absY,0.5,0)
    cv2.imshow('量子算法', dst)
    cv2.waitKey(0)

    dst = cv2.medianBlur(dst, 9)
    ret_1,ddst=cv2.threshold(dst,9,255,cv2.THRESH_BINARY)
    # 显示二值化处理图片
    cv2.imshow('ddstByAverge',ddst)
    cv2.waitKey(0)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray2 = cv2.dilate(ddst, element)
    output=cv2.dilate(gray2,element)
    cv2.imshow('dilate',output)
    cv2.waitKey(0)
    # 寻找图片中出现过得轮廓
    # im,contours,hierarchy=cv2.findContours(ddst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    # 在保存的轮廓里利用宽度和高度进行筛选
    # cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)
    # cv2.imwrite('./pic\\TelCamera\\frame.jpg', frame)
    alfa = 1
    img_new = []
    for c in contours:
        x , y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if w > 20 and h > 20:
            # print(w,h)
            rois.append((x, y, w, h))

            # 截取图片
            top, bottom, left, right = (0, 0, 0, 0)
            img = output[y:y+h,x:x+w]
            img = cv2.medianBlur(img, 9)
            # cv2.imshow('img_cut_BYAvarge',img)
            # cv2.waitKey(0)
            h, w = img.shape
            # print(img.shape)
            longest_edge = max(h, w)
            if h < longest_edge:
                dh = longest_edge - h
                top = dh // 2
                bottom = dh - top
                # print('longest_edge{} top{} bottom{}'.format(longest_edge,top,bottom))
            elif w < longest_edge:
                dw = longest_edge - w
                left = dw // 2
                right = dw - left
                # print('longest_edge{} left{} right{}'.format(longest_edge, left, right))
            else:
                # print('没有进判断')
                pass

            BLACK = [0, 0, 0]
            constant = cv2.copyMakeBorder(
                img, top+50, bottom+50, left+50, right+50, cv2.BORDER_CONSTANT, value=BLACK)
            cv2.imshow('square',constant)
            cv2.waitKey(0)
            img_28 = cv2.resize(constant,(28,28))
            # cv2.imshow('28*28', img_28)
            # cv2.waitKey(0)
            img_new.append(img_28)   #正方形图片

    # cv2.imshow('frame',frame)
    # cv2.waitKey(0)
    return frame,img_new,rois


# cv2.WINDOW_NORMAL
# image = cv2.imread('./pic\\TelCamera\\num_good_1.jpg')  # './pic\\TelCamera\\Num.jpg'  ./pic\\TelCamera\\mul_num.jpg
# num = where_num(image)








