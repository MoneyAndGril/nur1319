import random
import matplotlib.pyplot as plt
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QLabel, QWidget, QFileDialog, QFrame, \
    QListView
from PyQt5.QtGui import QPen, QPainter, QPixmap,QIcon
from PyQt5.QtCore import Qt, QPoint
from PyQt5 import QtCore, QtGui
from PyQt5 import QtWidgets
from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image
import os
import cv2
from PIL import Image
from Camera_Handle import where_num
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class MainWindows(QMainWindow):
    def __init__(self):
        super(MainWindows, self).__init__()
        self.InitUI()

    def InitUI(self):
        self.model = keras.models.load_model('./weights.131-0.9962.hdf5')
        (_, _), (self.test_images, self.test_labels) = mnist.load_data()   #获取mnist 测试集
        self.num = 0
        self.NeedHandle = False
        self.file = ''
        self.resize(500, 400)
        self.setFixedSize(500, 400);
        self.setWindowTitle('手写数字识别')

        self.button_regonize = QPushButton(self)
        self.button_regonize.setObjectName("button_regonize")
        self.button_regonize.setGeometry(73, 270, 70, 50)
        self.button_regonize.setText('识别')
        self.button_regonize.setStyleSheet("#button_regonize{\n"
                                           "    background-color:white;\n"
                                           "    color:black;\n"
                                           "    border:1px solid black;\n"
                                           "    border-radius:5px;\n"
                                           "}\n"
                                           "#button_regonize:hover{\n"
                                           "    background-color:black;\n"
                                           "    color:white\n"
                                           "}")
        # vbox.addWidget(self.button_regonize)
        # vbox.addStretch()

        self.button_clear = QPushButton(self)
        self.button_clear.setObjectName('button_clear')
        self.button_clear.setGeometry(176, 270, 70, 50)
        self.button_clear.setText('清空')
        self.button_clear.setStyleSheet("#button_clear{\n"
                                           "    background-color:white;\n"
                                           "    color:black;\n"
                                           "    border:1px solid black;\n"
                                           "    border-radius:5px;\n"
                                           "}\n"
                                           "#button_clear:hover{\n"
                                           "    background-color:black;\n"
                                           "    color:white\n"
                                           "}")
        # vbox.addWidget(self.button_clear)
        # vbox.addStretch()    # 什么意思

        self.result_text = QLabel(self)
        self.result_text.setText('结果为:')
        self.result_text.setGeometry(300, 160, 50, 50)

        self.result_lb = QLabel(self)
        self.result_lb.setText('')
        self.result_lb.setGeometry(300, 200, 91, 81)
        self.result_lb.setAlignment(Qt.AlignCenter)
        self.result_lb.setStyleSheet('font: 87 48pt "Arial Black";border:1px solid rgb(0,0,0)')
        # vbox.addWidget(self.result_lb)
        # vbox.addStretch()


        # self.label_border = QLabel(self)
        # self.label_border.setGeometry(50, 50, 200, 200)
        # self.label_border.setStyleSheet('border:2px solid black')

        self.lb = MyPaint_label(self)
        self.lb.setGeometry(50, 50, 200, 200)
        # self.lb.setStyleSheet('border:2px solid black')

        self.cb = QtWidgets.QComboBox(self)
        self.cb.addItem('手写数字识别')
        self.cb.addItem('样本识别')
        self.cb.setGeometry(300, 120, 150, 50)
        self.cb.setStyleSheet('QAbstractItemView #item{height:120px;}')
        self.cb.setView(QListView())
        # self.cb.setStyleSheet('border:1px black solid;'
        #                       'border-radius:5px')

        self.file_button = QPushButton(self)
        self.file_button.setObjectName('file_button')
        self.file_button.setText("选择图片")
        self.file_button.setStyleSheet("#file_button{\n"
                                           "    background-color:white;\n"
                                           "    color:black;\n"
                                           "    border:1px solid black;\n"
                                           "    border-radius:5px;\n"
                                           "}\n"
                                           "#file_button:hover{\n"
                                           "    background-color:black;\n"
                                           "    color:white\n"
                                           "}")
        self.file_button.setGeometry(300, 60, 80, 50)
        # self.file_button.setGeometry(QtCore.QRect(30, 20, 30, 30))

        self.button_clear.clicked.connect(lambda: self.clear_list())
        self.button_regonize.clicked.connect(lambda: self.regonize())
        self.file_button.clicked.connect(lambda:self.openFile())
        self.cb.activated.connect(lambda:self.current_comb())
        # self.cb.currentIndexChanged[str].connect(self.print_value)  # 在下拉列表中，鼠标移动到某个条目时发出信号，传递条目内容
        # self.cb.highlighted[int].connect(self.print_value)  # 在下拉列表中，鼠标移动到某个条目时发出信号，传递条目索引
    def random_pic(self):
        num = self.test_images.shape[0] # (10000,28,28)
        rand = random.randint(0, num-1)    # fname为列表 不能越界 最大值减一
        img = np.array(self.test_images[rand])

        cv2.imwrite('./pic\\icon\\rand_test.png',img)
        pic = QPixmap('./pic\\icon\\rand_test.png').scaled(self.lb.pixmap.width(), self.lb.pixmap.height())
        self.lb.pixmap = pic
        self.lb.update()

    def current_comb(self):
        if self.cb.currentText() == '手写数字识别':
            self.button_clear.setText('清空')
            self.result_lb.setText('')
            self.lb.pixmap.fill(Qt.black)
            self.lb.update()
        elif self.cb.currentText() == '样本识别':
            self.button_clear.setText('刷新')
            self.random_pic()

    def clear_list(self):
        #手写数字时 清空
        if self.cb.currentText() == '手写数字识别':
            self.result_lb.setText('')
            self.lb.pixmap.fill(Qt.black)
            self.lb.update()

        #样本识别时 刷新
        elif self.cb.currentText() == '样本识别':
            # print('self.random_pic()')
            self.random_pic()   # 出错卡顿
            # print('样本识别 刷新')

    def regonize(self):
        if self.NeedHandle == False:
            savePath = f'./pic\\save_recgonize\\1.png'
            self.lb.pixmap.save(savePath)   # 保存pixmap图片
            img = keras.preprocessing.image.load_img(savePath, target_size=(28, 28))
            img = img.convert('L')

            x = keras.preprocessing.image.img_to_array(img)
            x = np.reshape(x,[-1,28,28,1])
            x = x / 255.0
            prediction = self.model.predict(x)
            output = np.argmax(prediction, axis=1)
            # np.set_printoptions(precision=4)
            # print(output.shape)
            self.result_lb.setText(str(output[0]))
            present = prediction[0][output][0] * 100
            # a = prediction[0][output]
            print(f'\r识别结果: {output[0]} 相似度:{present:.2f}% {prediction[0][output][0]}')
        else:
            print('进入了')
            print(self.file)
            img = Image.open(self.file)
            img = np.asarray(img)
            pic,img_list,rec_list = where_num(img)
            img_list = np.array(img_list)
            x = img_list.reshape([-1,28,28,1])
            x = x / 255.0
            print(x.shape)
            if x.shape[0] == 0:
                print('None')
                return
            prediction = self.model.predict(x)
            print(prediction.shape)

            for i in range(prediction.shape[0]):
                x,y,w,h = rec_list[i]
                print(x,y,w,h)
                predict = np.argmax(prediction[i])
                print(prediction[i][predict])
                # if round(prediction[i][predict],5) <= round(0.5,5):
                #     continue
                # print('predict',predict,'predict_type',type(predict))
                predict = str(predict)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, predict, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
                print('ok')
            cv2.imshow('last',img)
            cv2.waitKey(0)
            cv2.imwrite('./pic\\write\\pic_regonized.png',img)
            self.NeedHandle = False

    def openFile(self):
        fname = QFileDialog.getOpenFileName(self, "选择图片文件", ".")
        if fname[0]:
            self.file = fname[0]
            self.NeedHandle = True
            pic = QPixmap(fname[0]).scaled(self.lb.pixmap.width(), self.lb.pixmap.height())
            print(fname[0])
            self.lb.pixmap = pic

class MyPaint_label(QLabel):
    def __init__(self,parent):
        super(MyPaint_label, self).__init__(parent)
        self.pixmap = QPixmap(200, 200)
        self.pixmap.fill(Qt.black)


        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        self.painter = QPainter()
        self.setMouseTracking(False)

    def paintEvent(self, event):
        self.painter.begin(self)
        self.painter.drawPixmap(0,0,self.pixmap)
        self.painter.end()

    def mousePressEvent(self, event):
        # if event.button() == Qt.LeftButton:
        self.lastPoint = event.pos()
        self.endPoint = self.lastPoint

    def mouseMoveEvent(self, event):  # 重写鼠标移动事件
        # if event.buttons() == Qt.LeftButton:
        self.endPoint = event.pos()

        self.painter.begin(self.pixmap)   # 16
        pen = QPen(Qt.white, 16, Qt.SolidLine)
        self.painter.setPen(pen)
        self.painter.drawLine(self.lastPoint,self.endPoint)
        self.painter.end()

        self.lastPoint = self.endPoint
        self.update()  # 更新绘图事件,每次执行update都会触发一次paintEvent(self, event)函数


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('./pic\\icon\\five_icon.png'))
    mainWindow = MainWindows()
    mainWindow.show()
    sys.exit(app.exec_())
