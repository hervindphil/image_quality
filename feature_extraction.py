import numpy as np
import cv2
from scipy.signal import convolve2d

#image object is 'img'
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#height and width
height, width,_    = img.shape

#distinct pixel rate
dist_pixel = len(np.unique([str(x) for x in img[:,:,:].reshape(-1, 3).tolist()]))
dist_pixel_rate = dist_pixel / float(height * width)


#luminance
luminance = img_gray.mean()

#blur
blur = cv2.Laplacian(img, cv2.CV_64F).var()

#sharpness
gy, gx = np.gradient(img_gray)
gnorm = np.sqrt(gx**2 + gy**2)
sharpness = np.average(gnorm)

#saturation
saturation = img_hsv[:,:,1].mean()

#rgb mean
red = img[:,:,2].mean()
green = img[:,:,1].mean()
blue = img[:,:,0].mean()

red_var = img[:,:,2].var()
green_var = img[:,:,1].var()
blue_var = img[:,:,0].var()
gray_var = img_gray.var()
