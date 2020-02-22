import cv2 as cv
from segmentator import Segmentator

import matplotlib.pyplot as plt

seg=Segmentator()
img=cv.imread('road.jpg')
opn,close=seg.morph(img)

cv.imshow('open',opn)
cv.imshow('close',close)
cv.waitKey()
cv.destroyAllWindows()
#plt.show()