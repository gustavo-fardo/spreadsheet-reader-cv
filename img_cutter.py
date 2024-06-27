import cv2 as cv
import numpy as np

def cut_outside(img:cv.Mat,percentage_cut,percentage_open):
    buffer = np.empty(img.shape,np.float32)
    buffer = img.copy()
    buffer = 255 - buffer
    rows, cols = buffer.shape
    hist_horizontal = np.zeros((cols,1),np.float32)
    sum_hist_horizontal = 0
    for i in range(0,cols):
        for j in range(0,rows):
            value = buffer[j,i]
            hist_horizontal[i]+= value
            sum_hist_horizontal += value
    
    hist_vertical = np.zeros((rows,1),np.float32)
    sum_hist_vertical = 0
    for i in range(0,rows):
        for j in range(0,cols):
            value = buffer[i,j]
            hist_vertical[i]+= value
            sum_hist_vertical += value

 
    left = 0
    right = cols
    up =0
    down = rows

    sum = 0
    for i in range(0, cols):
        sum+= hist_horizontal[i]
        if sum/sum_hist_horizontal > percentage_cut:
            left = i
            break
    
    sum = 0 
    for i in range(0, cols):
        sum+= hist_horizontal[i]
        if sum/sum_hist_horizontal > 1- percentage_cut:
            right = i
            break
    
    sum = 0
    for i in range(0, rows):
        sum+= hist_vertical[i]
        if sum/sum_hist_vertical > percentage_cut:
            up = i
            break
    
    sum = 0 
    for i in range(0, rows):
        sum+= hist_vertical[i]
        if sum/sum_hist_vertical > 1-percentage_cut:
            down = i
            break
    
    augmentx= (right-left)//2*percentage_open
    augmenty= (down-up)//2*percentage_open
    if(left-augmentx<0):
        left = 0 
    else:
        left = left-augmentx
    if(up-augmenty<0):
        up = 0 
    else:
        up = up-augmenty

    if(right+augmentx>=cols):
        right = cols-1
    else:
        right = right+augmentx

    if(down+augmenty>=rows):
        down = rows-1
    else:
        down = down+augmenty

    return int(left),int(right),int(up),int(down)