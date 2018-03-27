import cv2
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')

def image_test():
    # image part
    print(cv2.__version__)
    img = cv2.imread("./a1p5.jpeg", 0)
    # imread FLAG
    # 1: cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
    # 0: cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
    # -1: cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    cv2.imwrite('./output.jpeg', img)
    cv2.imshow('image', img)
    k = cv2.waitKey(0)
    if k == 27 or k == ord('q'):  # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('messigray.png', img)
    cv2.destroyAllWindows()


# video part
def video_test():
    cap = cv2.VideoCapture('test.mp4')

    while (cap.isOpened()):
        ret, frame = cap.read()
        # print(frame.size)
        # print(frame[0,:][:])
        if frame is None:
            # try to sampling
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(frame[0,:][:])
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()



if __name__ == '__main__':
    # image_test()
    video_test()