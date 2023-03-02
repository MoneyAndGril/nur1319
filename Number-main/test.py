# test randint  包含边界
import cv2
from PIL import Image
# path = 'D:/U盘/Tool/nur-master/nur-master/Number-main/pic/TelCamera/22.jpg'
# path = 'D:\\U盘\\Tool\\nur-master\\nur-master\\Number-main\\pic\\TelCamera\\22.jpg'
# # path = 'pic/TelCamera/22.jpg'
# img = cv2.imread(path)
# cv2.imshow('img', img)
#
# cv2.waitKey(0)

# fname = r'D:/U盘/Tool/nur-master/nur-master/Number-main/pic/HandDraw/4.png'
# img = cv2.imread(fname)
# # print(img)
# cv2.imshow('img',img)
# cv2.waitKey(0)

fname = r'D:/U盘/Tool/nur-master/nur-master/Number-main/pic/HandDraw/4.png'
pic = Image.open(fname)
pic.show('handle')

if round(0.628888888,5) <= round(0.7,5):
    print('ok')