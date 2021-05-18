import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

path = ""
img = cv.imread("")
img_blur = cv.bilateralFilter(img,5,150,150)
#cv2.imshow(img_blur)
hsv = cv.cvtColor(img_blur, cv.COLOR_BGR2HSV)
low_blue = np.array([55, 0, 0])
high_blue = np.array([118, 255, 255])
mask = cv.inRange(hsv, low_blue, high_blue)
#cv2.imshow(mask)
seg = cv.bitwise_and(img,img, mask= mask)
image = seg
# 构建图像数据
data = image.reshape((-1,3))
data = np.float32(data)
# MAX_ITER最大迭代次数，EPS最高精度
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
num_clusters = 2
ret,label,center=cv.kmeans(data, num_clusters, None, criteria, num_clusters, cv.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
# 颜色label
color = np.uint8([[255, 0, 0],
                  [141, 88, 75],
                  [128, 128, 128],
                  [0, 255, 0],
                  [64,64,64],])

res = color[label.flatten()]
print(res.shape)
# 显示
result = res.reshape((image.shape))
cv2_imshow(result)
final = cv.bitwise_and(img_blur,result, mask=mask)
cv.waitKey(0)
cv.destroyAllWindows()
