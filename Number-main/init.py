import os

Commend = {'numpy':'pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy',
    'tesorflow':'pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==2.6',
    'keras':'pip install -i https://pypi.tuna.tsinghua.edu.cn/simple keras==2.6.0',
    'pillow':'pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow',
    'PyQt5':'pip install -i https://pypi.tuna.tsinghua.edu.cn/simple PyQt5==5.15.1',
    'pyqt5-tools':'pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyqt5-tools==5.15.1.2',
    'opencv':'pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python',
    'protobuf':'pip install -i https://pypi.tuna.tsinghua.edu.cn/simple protobuf==3.20.0'
}

for i,commend in Commend.items():
    res = os.system(commend)
    if res == 0:
        print(str(i)+'安装成功')










