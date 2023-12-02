import cv2
import numpy as np

#method for converting images to uint8
def convertUnit8(img):
  img2 = np.zeros_like(img)
  cv2.normalize(img, img2, 0, 255, cv2.NORM_MINMAX)
  img2 = np.uint8(img2)
  return img2


def image_subtraction(img1,img2):
    #convert images to uint8 as descibed in the paper
    img1 = img1[:180][:]
    #img2 = img2[:][:280]
    print(img1.shape)
    print(img2.shape)
    img3 = convertUnit8(img1)
    img4 = convertUnit8(img2)
    return  cv2.subtract(img3,img4)

def binarization(img):
    #convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(gray,(5,5),0)
    ret,threshold = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return threshold


img = cv2.imread("object2.jpg",1)
img2 = cv2.imread("images.jpeg",1)
img3 = binarization(image_subtraction(img,img2))
cv2.imshow("window1",img)
cv2.waitKey(5000)
cv2.destroyAllWindows()
cv2.imshow("window2",img3)
cv2.waitKey(5000)
cv2.destroyAllWindows()