import cv2
def IpCamera(video_ip = 'http://admin:admin@10.3.242.51:8081/video'):
    cv2.namedWindow('camera', 1)

    # 摄像头 参数0 表示调用内置摄像头 电脑自己的摄像头
    # cv2.VideoCapture(0)

    # ip 摄像头 局域网地址  手机应用
    video_ip = 'http://admin:admin@10.3.242.51:8081/video'
    capture = cv2.VideoCapture(video_ip)

    while True:
        success, img = capture.read()
        cv2.imshow('camera',img)

        # 按键处理
        key = cv2.waitKey(1)
        # esc == 27
        if key == 27:
            # esc
            break
        if key == 32:
            # 32 空格 相等于 模拟 键盘空格拍照
            filename = 'frame.jpg'
            cv2.imwrite(filename, img)

    # 释放摄像头
    capture.release()
    # 关闭窗口
    cv2.destroyWindow('camera')  # 指定释放 哪个窗口

if __name__ == '__main__':
    IpCamera()

