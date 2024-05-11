import cv2
import matplotlib.pyplot as plt

def hisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img


fname='./sample/architecture1-Hekm.png'
img=cv2.imread(fname)

# cv2.imshow('img', img)
img2=hisEqulColor(img)
cv2.imwrite("res.png", img2)
# plt.imshow(img2)
# plt.show()
# a = input()