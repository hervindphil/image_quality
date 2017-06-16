import numpy as np
import cv2
from scipy.signal import convolve2d

#image object named 'img' is a numpy array with pixel value is BGR (opencv default)


#1. grayscale rgb mean and variance
def get_pixel_summary(img):
  img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
  luminance = img_gray.mean()
  red = img[:,:,2].mean()
  green = img[:,:,1].mean()
  blue = img[:,:,0].mean()
  
  gray_var = img_gray.var()
  red_var = img[:,:,2].var()
  green_var = img[:,:,1].var()
  blue_var = img[:,:,0].var()

  return luminance, red, green, blue, gray_var, red_var, green_var, blue_var

#2. distinct pixel rate
def get_dist_pixel_rate(img):
  height, width,_    = img.shape
  dist_pixel = len(np.unique([str(x) for x in img[:,:,:].reshape(-1, 3).tolist()]))
  dist_pixel_rate = dist_pixel / float(height * width)
  return dist_pixel_rate

#3. contrast
def get_contrast(img):
  img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  contrast = img_gray.max()-img_gray.min()
  return contrast
  
#4. saturation
def get_saturation(img):
  img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  saturation = img_hsv[:,:,1].mean()
  return saturation

#5 blur
def get_blur(img):
  blur = cv2.Laplacian(img, cv2.CV_64F).var()
  return blur

#6 sharpness
def get_sharpness(img):
  gy, gx = np.gradient(img_gray)
  gnorm = np.sqrt(gx**2 + gy**2)
  sharpness = np.average(gnorm)
  return sharpness



