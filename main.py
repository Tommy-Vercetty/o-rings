import cv2 as cv
import numpy as np
import time

def threshold(img, thresh):
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if img[x,y] > thresh:
                img[x,y] = 255
            else:
                img[x,y] = 0
    return img

for i in range(1, 16):
    #read in an image into memory
    img = cv.imread('c:/users/tommy/Downloads/Orings/Oring' + str(i) + '.jpg',0)
    thresh = 100
    bw = threshold(img, thresh)
    rgb = cv.cvtColor(bw, cv.COLOR_GRAY2RGB)
    #Annotating the image. We are adding the word Hello in colour blue on the image
    cv.putText(rgb, "Image: " + str(i), (40, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #Annotating the image with a circle.
    cv.circle(rgb, (40, 40), 20, (0, 0, 255))
    #We can show the image using the OpenCV open function
    cv.imshow('thresholded image', rgb)
    cv.waitKey(0)
    cv.destroyAllWindows()
