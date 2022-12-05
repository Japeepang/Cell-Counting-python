import cv2 
import numpy as np


img = cv2.imread('macrocytosis.jpg')
img = cv2.resize(img, (400,400))
gray = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
# blurred = cv2.bilateralFilter(gray,3,21,21)

# laplacian
lap = cv2.Laplacian(blurred, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))

# Sobel
sobelX = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobelCombined = cv2.bitwise_or(sobelX, sobelY)

# Canny
canny = cv2.Canny(blurred, 30, 150)


# Laplacian+Contours
cnts, _ = cv2.findContours(lap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
lap_img = img.copy()
cv2.drawContours(lap_img, cnts, -1, (0,255,0),2)
# print('Lap: ', len(cnts))

# Sobel+Contours
cnts, _ = cv2.findContours(sobelCombined.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sobel_img = img.copy()
cv2.drawContours(sobel_img, cnts, -1, (0,255,0),2)
# print('Sobel: ', len(cnts))


# Canny+Contours
cnts, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
canny_img = img.copy()
cv2.drawContours(canny_img, cnts, -1, (0,255,0),2)

i = 1
for i in range(len(cnts)):
    # (xL, xU)
    xs = [x for [(x,y)] in cnts[i]]
    xs.sort()
    xL, xU = xs[0], xs[-1]    
    # (yL, yU)
    ys = [y for [(x,y)] in cnts[i]]
    ys.sort()
    yL, yU = ys[0], ys[-1]    
    # bounding box
    canny_img = cv2.rectangle(canny_img, (xL,yL), (xU, yU), (0,0,255),2)
    # cell number 
    cv2.putText(canny_img, str(i), (xL, yL-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)


# cv2.imshow('img', img)
# cv2.imshow('Laplacian', lap)
# cv2.imshow('Laplacian Image', lap_img)
# cv2.imshow('Sobel', sobelCombined)
    #cv2.imshow('Sobel Image', sobel_img)
# cv2.imshow('Canny', canny)
    cv2.imshow('Canny Image', canny_img)
    cv2.waitKey(1000)