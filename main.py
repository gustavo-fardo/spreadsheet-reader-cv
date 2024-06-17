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

INPUT_IMAGE =  'out-scaled-down.png'
KSIZE_ADAPTIVE_THRESHOLD = 21
C_ADAPTIVE_THRESHOLD = 10
ALT_ENTRE_LINHAS = 5
KERNEL_EROSAO = (7, 7)
KERNEL_DILATACAO = (7, 7)
KERNEL_EROSAO_HORIZONTAL = (2, 5)
MARGIN_ANGLES = 5
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
    img_or = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img_or is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()
    height, width = img_or.shape[0], img_or.shape[1]

    draw_col = img_or.copy()

    img_or = cv2.cvtColor(img_or, cv2.COLOR_BGR2GRAY)

    # Binarizacao 
    blur = img_or
    img_bin = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, KSIZE_ADAPTIVE_THRESHOLD, C_ADAPTIVE_THRESHOLD)
   
    # Converte para float
    img_bin = img_bin.reshape ((img_bin.shape [0], img_bin.shape [1], 1))
    img_bin = img_bin.astype (np.float32)/255

    # Fechamento
    kernel_erosao = np.ones(KERNEL_EROSAO, np.uint8) 
    img_fechamento = cv2.erode(img_bin, kernel_erosao, iterations=1)
    kernel_dilatacao = np.ones(KERNEL_DILATACAO, np.uint8) 
    preproc= cv2.dilate(img_fechamento, kernel_dilatacao, iterations=1)
    # preproc = img_bin
    cv2.imwrite ('_1-preprocessada.png', preproc*255)

    # Canny
    canny = (preproc).astype(np.uint8)*255
    canny = cv2.Canny(canny, 50, 150, apertureSize=3)
    cv2.imwrite ('_2-canny.png', canny)

    # Linhas Hough Infinitas
    canvas1 = np.zeros((height, width), dtype=np.uint8)
    lines = cv2.HoughLines(canny, 1, np.pi/180, 300)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            angle = theta * 180 / np.pi
            if abs(angle) < MARGIN_ANGLES or abs(angle-90) < MARGIN_ANGLES or abs(angle - 180) < MARGIN_ANGLES or abs(angle - 270) < MARGIN_ANGLES:
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + width*(-b)), int(y0 + height*(a)))
                pt2 = (int(x0 - width*(-b)), int(y0 - height*(a)))
                cv2.line(canvas1, pt1, pt2, (255,0,0), 3, cv2.LINE_AA)
    canvas1 = cv2.dilate(canvas1, kernel_dilatacao, iterations=1)

    # Linhas Hough Finitas
    canvas2 = np.zeros((height, width), dtype=np.uint8)
    lines = cv2.HoughLinesP(
            canny, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=300, # Min number of votes for valid line
            minLineLength=width//1000+2, # Min allowed length of line
            maxLineGap=width//2 # Max allowed gap between line for joining them
            )
    for points in lines:
        x1,y1,x2,y2=points[0]
        cv2.line(canvas2,(x1,y1),(x2,y2),(255,255,255),2)
    canvas2 = cv2.dilate(canvas2, kernel_dilatacao, iterations=1)
    
    # Linhas detectadas
    det_lines = canvas1 & canvas2
    det_lines = cv2.erode(det_lines, kernel_erosao, iterations=1)
    condition = det_lines > 0.5
    cv2.imwrite ('_3.1-canvas.png', canvas1)
    cv2.imwrite ('_3.2-canvas.png', canvas2)
    cv2.imwrite ('_3-canvas.png', det_lines)

    # Desenha imagem de saída
    draw_col[..., 0] = np.where(condition, 255, draw_col[..., 0])  # Red channel
    draw_col[..., 1] = np.where(condition, 0, draw_col[..., 1])    # Green channel
    draw_col[..., 2] = np.where(condition, 0, draw_col[..., 2])    # Blue channel

    # "Linhas invisiveis" verticais
    # pontos_linhas = linhasPorEspaco(preproc)
    # for p in pontos_linhas:
    #     cv2.circle(draw_col, p, radius=0, color=(0, 255, 0), thickness=-1)

    # # Separa componentes
    # erosao_horizontal = c
    # componentes = contaComponentes(img)
    # tabela = []
    # height_line = -ALT_ENTRE_LINHAS
    # row = []

    # # Arranja componentes na estrutura da tabela
    # for c in componentes:
    #     crop = img_or[c['B']:c['T'], c['L']:c['R']]
    #     crop = np.pad(crop, ((PAD_SIZE, PAD_SIZE), (PAD_SIZE, PAD_SIZE)), mode='constant', constant_values=255)
    #     text = pytesseract.image_to_string(crop)
    #     text = ''.join(ch for ch in text if ch.isprintable())
    #     y_comp_middle = (c['T']-c['B'])//2+c['B']
    #     if(y_comp_middle < (height_line + ALT_ENTRE_LINHAS) and y_comp_middle > (height_line - ALT_ENTRE_LINHAS)):
    #         row.append(text)
    #     else:
    #         if row != []:
    #             tabela.append(row)
    #         row = [text]
    #     height_line = y_comp_middle
    #     cv2.rectangle(draw_col, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,255))
    # tabela.append(row)

    # Transforma em csv
    filename = INPUT_IMAGE[:-4] + ".csv"
    # with open(filename, 'w') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerows(tabela)

    # c = componentes[83]
    # cv2.imwrite("crop.png", crop)

    cv2.imwrite ('_4-componentes.png', draw_col)
    
    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()

#===============================================================================