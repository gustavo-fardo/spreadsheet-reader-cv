#===============================================================================
# Trabalho 5 - Chroma Key
#-------------------------------------------------------------------------------
# Autor: Gustavo Fardo Armênio
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import numpy as np
import cv2
import math
import csv
import pytesseract

# increase the recursion limit for floodfill
sys.setrecursionlimit(10**6)

#===============================================================================

INPUT_IMAGE =  '2.bmp'
ALT_ENTRE_LINHAS = 5
THRESH_BIN = 0.8
KERNEL_EROSAO = (2, 5)
PAD_SIZE = 50

def floodFill(img, y, x, componente, shape):
    ''' Realiza o flood fill de um componente de uma imagem binarizada

Parâmetros: img: imagem de entrada. 
            y, x: coordenadas do pixel.
            componente: componente atual.
            shape: tamanho da imagem.
            
Valor de retorno: nenhum, somente para voltar na recursão.'''
    if not(y in range(shape[0]) and x in range(shape[1])) or img[y, x] != 0:
        return
    img[y, x] = componente["n_pixels"]
    componente["n_pixels"] += 1
    componente["L"] = x if x < componente["L"] else componente["L"] 
    componente["R"] = x if x > componente["R"] else componente["R"] 
    componente["T"] = y if y > componente["T"] else componente["T"] 
    componente["B"] = y if y < componente["B"] else componente["B"] 
    floodFill(img, y+1, x, componente, shape)
    floodFill(img, y-1, x, componente, shape)
    floodFill(img, y, x+1, componente, shape)
    floodFill(img, y, x-1, componente, shape)

def contaComponentes(img):
    ''' Marca os componentes conexos de uma imagem binarizada a partir do floodfill

Parâmetros: img: imagem de entrada. 
            
Valor de retorno: dicionário dos componentes com labels, numero de pixels e coordenadas das extremidades.'''
    componentes = []
    label_atual = 1
    shape = np.shape(img)
    for y in range(shape[0]):
        for x in range(shape[1]):
            if(img[y, x] == 0):
                componente = {"label": label_atual, "n_pixels": 0, "T": 0, "B": shape[0], "L": shape[1], "R": 0}
                floodFill(img, y, x, componente, shape)
                componentes.append(componente)
                label_atual += 1
    return componentes

def linhasPorEspaco(img):
    pontos = []
    h, w = img.shape[0], img.shape[1]
    for y in range(1, h):
        last_border = [0,0,0] # Value x-1, value x, x
        curr_border = [0,0,0] # Value x-1, value x, x
        for x in range(1, w):
            curr_pair = [img[y, x-1], img[y, x], x]
            if curr_pair[0] != curr_pair[1]:
                last_border = curr_border
                curr_border = curr_pair
                if curr_border[:-1] == [1,0] and last_border[:-1] == [0,1]:
                    pontos.append([((curr_border[2]-last_border[2])//2)+last_border[2], y])
    return pontos

def main ():
    # Abre a imagem em escala de cinza.
    img_or = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img_or is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    img_or_col = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img_or_col is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    #img_or_col = img_or_col[:,:-50]

    draw_col = img_or_col.copy()

    img = img_or.reshape ((img_or.shape [0], img_or.shape [1], 1))
    img = img_or.astype (np.float32)/255

    # img = cv2.GaussianBlur(img, (15,1), 0)
    # img = cv2.medianBlur(img, 5)

    # Binarizacao
    img_bin = np.float32(np.where(img > THRESH_BIN, 1, 0))

    # Erosao
    kernel_erosao = np.ones(KERNEL_EROSAO, np.uint8) 
    img = cv2.erode(img_bin, kernel_erosao, iterations=2)

    pontos_linhas = linhasPorEspaco(img)

    cv2.imwrite ('preprocessada.png', img*255)

    # Linhas Hough
    lines_img = (1-img_bin).astype(np.uint8)*255
    # lines_img = cv2.Canny(draw_col, 50, 200, None, 3)
    cv2.imwrite ('canny.png', lines_img)
    lines = cv2.HoughLines(lines_img, 1, np.pi / 180, 200)
 
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            angle = theta * 180 / np.pi
            # Check for horizontal lines (angle close to 0 or 180 degrees)
            if abs(angle) < 1 or abs(angle - 180) < 1:
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(draw_col, pt1, pt2, (255,0,0), 3, cv2.LINE_AA)

    # Linhas verticais
    for p in pontos_linhas:
        cv2.circle(draw_col, p, radius=0, color=(0, 255, 0), thickness=-1)

    # Separa componentes
    componentes = contaComponentes(img)
    tabela = []
    height_line = -ALT_ENTRE_LINHAS
    row = []

    # Arranja componentes na estrutura da tabela
    for c in componentes:
        crop = img_or[c['B']:c['T'], c['L']:c['R']]
        crop = np.pad(crop, ((PAD_SIZE, PAD_SIZE), (PAD_SIZE, PAD_SIZE)), mode='constant', constant_values=255)
        text = pytesseract.image_to_string(crop)
        text = ''.join(ch for ch in text if ch.isprintable())
        y_comp_middle = (c['T']-c['B'])//2+c['B']
        if(y_comp_middle < (height_line + ALT_ENTRE_LINHAS) and y_comp_middle > (height_line - ALT_ENTRE_LINHAS)):
            row.append(text)
        else:
            if row != []:
                tabela.append(row)
            row = [text]
        height_line = y_comp_middle
        cv2.rectangle(draw_col, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,255))
    tabela.append(row)

    # Transforma em csv
    filename = INPUT_IMAGE[:-4] + ".csv"
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(tabela)

    c = componentes[83]
    cv2.imwrite("crop.png", crop)

    cv2.imwrite ('componentes.png', draw_col)
    
    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()

#===============================================================================