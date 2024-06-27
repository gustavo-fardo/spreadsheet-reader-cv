import cv2
import numpy as np
import math
import radon
import img_cutter

# saida = output, radon, binarized
def skew_corrector(src: cv2.Mat, radon_res = 300, angle_width = 90, angle_offset = -45, blur_kernel = 201, erosion_kernel = 10, percentage_cut = 0.15, percentage_open = 0.6):
    rows, cols,channels = src.shape
    img_blur = np.empty((rows,cols,channels),np.uint8)
    img_paper = np.empty((rows,cols,channels),np.uint8)
    img_blur = cv2.blur(src,(blur_kernel,blur_kernel)) #blur para remover o texto
    img_hsv = np.empty((rows,cols,channels),np.uint8)
    img_hsv = cv2.cvtColor(img_blur,cv2.COLOR_BGR2HSV)

    img_background = np.empty((rows,cols),np.uint8)

    img_s = np.empty((rows,cols),np.uint8)
    img_v = np.empty((rows,cols),np.uint8)

    #binariza imagem pela saturacao e pelo value
    ret, img_s = cv2.threshold(img_hsv[:,:,1],0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ret, img_v = cv2.threshold(img_hsv[:,:,2],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #cria mascara do papel
    img_background = (img_s/255)*(img_v/255)

    img_paper = 255 - src
    img_paper[:,:,0] = (img_paper[:,:,0]*img_background)
    img_paper[:,:,1] = (img_paper[:,:,1]*img_background)
    img_paper[:,:,2] = (img_paper[:,:,2]*img_background)
    img_paper = 255 - img_paper

    img_output = np.empty(img_background.shape)
    img_paper = cv2.normalize(img_paper, img_paper, alpha = 0, beta = 255, norm_type=cv2.NORM_MINMAX)
    img_output = cv2.cvtColor(img_paper,cv2.COLOR_BGR2GRAY)

    img_output = cv2.adaptiveThreshold(img_output,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,10)
    img_background = cv2.erode(img_background,np.ones((erosion_kernel,erosion_kernel)))
    img_output = img_output*img_background+(1-img_background)*255
    img_background = 1 - img_output/255

    radon_saida = np.empty((radon_res,radon_res,1),np.float32)

    for i in range(0,radon_res):
        radon_saida[:,i] = radon.radon(math.pi/180*(angle_offset  + i*angle_width/radon_res),img_background,300)
    fitness = radon.fitness_function(radon_saida)
    angle = radon.find_max(fitness)
    angle = angle/radon_res*angle_width + angle_offset
    img_output = radon.rotate_image(255-img_output,angle)
    img_output = 255-img_output

    l,r,u,d = img_cutter.cut_outside(img_output, percentage_cut, percentage_open)

    #usa buffer do blur para a imagem final
    img_blur = radon.rotate_image(src,angle)
    img_blur = img_blur[u:d,l:r]

    return img_blur, radon_saida, img_background