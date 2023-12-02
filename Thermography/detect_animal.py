import numpy as np
import cv2
import os

IMG_SIZE = (256,256)
THRESHOLD = int((256*256) * 0.009)

# method for converting images to uint8 and reshape
def convertUnit8(img):
  img2 = np.zeros_like(img)
  cv2.normalize(img, img2, 0, 255, cv2.NORM_MINMAX)
  img2 = np.uint8(img2)
  img2 = cv2.resize(img2,IMG_SIZE)
  return img2

def segment_warm(img):
    img = convertUnit8(img)
    lowRed = cv2.inRange(img,(0,50,50),(10,255,255))
    highRed = cv2.inRange(img,(160,50,50),(190,255,255))
    red = lowRed + highRed
    #yellow = cv2.inRange(img, (0, 100, 100), (0, 255, 255))
    orange = cv2.inRange(img, (5, 50, 50), (25, 255, 255))
    seg_img = red + orange
    return seg_img

def animal_exist(seg_img):
    pixels = np.reshape(seg_img,(IMG_SIZE[0]*IMG_SIZE[1]))
    red_pixels = np.sum([1 for i in pixels if i == 255])
    return red_pixels > THRESHOLD

if __name__ == '__main__':
    imgs = [img for img in os.listdir() if img.endswith('.jpg')]
    for i,path in enumerate(imgs):
        img = cv2.imread(path)
        red_seg = segment_warm(img)
        img = convertUnit8(img)
        seg_3_channel = cv2.cvtColor(red_seg, cv2.COLOR_GRAY2BGR)
        horizontal_imgs = np.hstack((img, seg_3_channel))
        cv2.imshow("img",horizontal_imgs); cv2.waitKey(1)
        exist = animal_exist(red_seg)
        path = "result"+str(i)+"_"+str(exist)+".png"
        cv2.imwrite("results/"+path,horizontal_imgs)
        print("Animal detected : ",exist)
