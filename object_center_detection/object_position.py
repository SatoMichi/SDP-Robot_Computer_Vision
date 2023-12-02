import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from object_segmentation import IMG_SIZE, segment_object, get_main_object

IMG_SIZE = (256,256)
X_CENTER = int(256//2)
IGNORE_GAP = 10

x_i, y_i = -1, -1

def click_event(event,x,y,flags,params):
    global x_i, y_i
    if event == cv2.EVENT_LBUTTONDOWN:
        x_i, y_i = x, y

def clickPoint(img):
    img = cv2.imread(img)
    img = cv2.resize(img, IMG_SIZE)
    cv2.imshow("window",img)
    cv2.setMouseCallback("window",click_event)
    cv2.waitKey(5000)
    print("Click : [",x_i," ",y_i,"]")
    


def get_object_center(img):
    M = cv2.moments(img)
    x = int(M['m10'] / M['m00'])
    y = int(M['m01'] / M['m00'])
    return np.array([x, y])

# get the gap betweeb center of the detected object and center of the camera image
def get_X_gap(baseImg,cameraImg):
    img_seg = segment_object(baseImg,cameraImg)
    img_obj = get_main_object(img_seg)
    center = get_object_center(img_obj)
    print("Center : ", center)
    gap = center[0] - X_CENTER

    showImg = cv2.imread(cameraImg)
    showImg = cv2.resize(showImg,IMG_SIZE)
    showImg[:,X_CENTER-5:X_CENTER+5,0] = np.ones_like(showImg[:,X_CENTER-5:X_CENTER+5,0])
    showImg = cv2.circle(showImg,(center[0],center[1]),radius=4,color=(255,0,0),thickness=-1)
    plt.title("Center of the object and offset line")
    plt.imshow(showImg)
    plt.show()

    return gap

def left_forward_right(gap):
    if gap < 0 and abs(gap) > IGNORE_GAP:
        return "turn_left"
    elif gap > 0 and abs(gap) > IGNORE_GAP:
        return "turn_right"
    else:
        return "go_forward"

if __name__ == "__main__":
    imgb = sys.argv[1]
    imgo = sys.argv[2]
    clickPoint(imgo)
    gap = get_X_gap(imgb,imgo)