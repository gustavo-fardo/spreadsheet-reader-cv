import cv2 as cv
import numpy as np
import math


def radon(theta,img:cv.Mat,resolution): 
    rows, cols = img.shape
    radius = math.sqrt(rows*rows+cols*cols)/2.0
    dist = radius*2.0/resolution
    directionx =(math.cos(theta))
    directiony =(math.sin(theta)) 
    originx = radius*(directionx-directiony)+ cols/2.0
    originy = radius*(directiony-directionx)+ rows/2.0

    radon_theta:cv.Mat = np.empty((int(resolution),1),np.float32)
    for i in range(0,int(resolution)):
        startx = originx+dist*directiony*i
        starty = originy+dist*directionx*i
        radon_theta[i]=0
        for j in range(0,int(resolution)):
            _x = startx -j*directionx*dist
            _y = starty - j*directiony*dist
            if(_x >= 0 and int(_x)< cols and _y>=0 and int(_y)<rows):
                radon_theta[i]+= bilinear_interpolation(img,_x,_y)/dist

    return radon_theta


#retirado de https://www.askpython.com/python-modules/numpy/bilinear-interpolation-python, creditos a Vignya Durvasula
def bilinear_interpolation(img, x, y):
    rows, cols = img.shape
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + 1
    y2 = y1 + 1
    if x2 >= cols:
        x2 = x1
    if y2 >= rows:
        y2 = y1
 
    p11 = img[y1, x1]
    p12 = img[y2, x1]
    p21 = img[y1, x2]
    p22 = img[y2, x2]
 
    x_diff = x - x1
    y_diff = y - y1
 
    interpolated = (p11 * (1 - x_diff) * (1 - y_diff) +
                          p21 * x_diff * (1 - y_diff) +
                          p12 * (1 - x_diff) * y_diff +
                          p22 * x_diff * y_diff)
 
    return interpolated



def fitness_function(radon:cv.Mat):
    cv.blur(radon,(3,3),radon)
    rows,cols,c = radon.shape
    fitness = np.empty((cols,1),np.float32)
    for i in range(0,cols):
        fitness[i]= radon[0,i]
        for j in range(1,rows):
            dy = -radon[j-1,i]+radon[j,i]
            fitness[i]+= math.fabs(dy)
    return fitness

def find_max(fitness:cv.Mat):
    rows, cols = fitness.shape
    max= 0

    for i in range(1,rows):
        if(fitness[i]>fitness[max]):
            max = i
    return max
            
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result