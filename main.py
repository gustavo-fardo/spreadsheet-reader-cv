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

INPUT_IMAGE =  './radon/1.png'
# Preprocessamento
KSIZE_ADAPTIVE_THRESHOLD = 21
C_ADAPTIVE_THRESHOLD = 10
KERNEL_EROSAO = (7, 7)
KERNEL_DILATACAO = (7, 7)
# Deteccao de linhas
HOUGH_THRESHOLD = 200
MAX_LINE_DIST = 20
MARGIN_ANGLES = 1
# Regressao de linhas invisiveis
PHOUGH_THRESHOLD = 200
KERNEL_EROSAO_HORIZONTAL = (5, 25)
KERNEL_DILATACAO_VERTICAL = (5, 1)
LINE_REPLACE_VALUE = 2 # De preferencia, menor que 0 ou maior que 1, nunca igual a 0 ou 1
MAX_X_DIST = 20
MAX_Y_DIST = 5
MIN_SUPPORT_INV_LINE = 200
#
ALT_ENTRE_LINHAS = 5
PAD_SIZE = 50

def prepocessamento(img_or):
    # Binarizacao 
    img_bin = cv2.adaptiveThreshold(img_or, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, KSIZE_ADAPTIVE_THRESHOLD, C_ADAPTIVE_THRESHOLD)
   
    # Converte para float
    img_bin = img_bin.reshape ((img_bin.shape [0], img_bin.shape [1], 1))
    img_bin = img_bin.astype (np.float32)/255

    # Fechamento
    kernel_erosao = np.ones(KERNEL_EROSAO, np.uint8) 
    img_fechamento = cv2.erode(img_bin, kernel_erosao, iterations=1)
    kernel_dilatacao = np.ones(KERNEL_DILATACAO, np.uint8) 
    preproc= cv2.dilate(img_fechamento, kernel_dilatacao, iterations=1)

    return preproc

def reduz_linhas(lines, max_line_dist):
    lines.sort()
    i_ant = lines[0]
    ponto_prox = [lines[0]]
    ponto_out = []
    for i in range(1, len(lines)):
        if lines[i][0] - i_ant[0] < max_line_dist:
            ponto_prox.append(lines[i])
        else:
            ponto_out.append(np.mean(ponto_prox, axis=0).tolist())
            ponto_prox = [lines[i]]
        i_ant = lines[i]
    ponto_out.append(np.mean(ponto_prox, axis=0).tolist())
    return ponto_out

def encontra_linhas_desenhadas(canny, draw_img, width, height):
    # Linhas Hough Infinitas
    det_lines = np.zeros((height, width), dtype=np.uint8)
    vertical_lines = []
    horizontal_lines = []
    lines = cv2.HoughLines(canny, 1, np.pi/180, HOUGH_THRESHOLD)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            angle = theta * 180 / np.pi
            if abs(angle) < MARGIN_ANGLES or abs(angle - 180) < MARGIN_ANGLES:
                vertical_lines.append((rho, theta))
            if abs(angle - 90) < MARGIN_ANGLES or abs(angle - 270) < MARGIN_ANGLES:
                horizontal_lines.append((rho, theta))

    # Afina linhas
    if vertical_lines != []:
        vertical_lines = reduz_linhas(vertical_lines, MAX_LINE_DIST)
    for rho, theta in vertical_lines:
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + width*(-b)), int(y0 + height*(a)))
        pt2 = (int(x0 - width*(-b)), int(y0 - height*(a)))
        cv2.line(det_lines, pt1, pt2, (255,0,0), 1, cv2.LINE_AA)

    if horizontal_lines != []:
        horizontal_lines = reduz_linhas(horizontal_lines, MAX_LINE_DIST)
    for rho, theta in horizontal_lines:
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + width*(-b)), int(y0 + height*(a)))
        pt2 = (int(x0 - width*(-b)), int(y0 - height*(a)))
        cv2.line(det_lines, pt1, pt2, (255,0,0), 1, cv2.LINE_AA)

    # Desenha imagem de saída com as linhas detectadas
    condition = det_lines > 0.5
    draw_img[..., 0] = np.where(condition, 255, draw_img[..., 0])  # Red channel
    draw_img[..., 1] = np.where(condition, 0, draw_img[..., 1])    # Green channel
    draw_img[..., 2] = np.where(condition, 0, draw_img[..., 2])    # Blue channel

    return det_lines, draw_img

def remove_linhas_desenhadas(canny, preproc, height, width):
    # Linhas Hough Finitas
    canvas2 = np.zeros((height, width), dtype=np.uint8)
    lines = cv2.HoughLinesP(
            canny, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=PHOUGH_THRESHOLD, # Min number of votes for valid line
            minLineLength=width//1000+2, # Min allowed length of line
            maxLineGap=width//2 # Max allowed gap between line for joining them
            )
    for points in lines:
        x1,y1,x2,y2=points[0]
        cv2.line(canvas2,(x1,y1),(x2,y2),(255,255,255),2)
    kernel_dilatacao = np.ones(KERNEL_DILATACAO, np.uint8) 
    canvas2 = cv2.dilate(canvas2, kernel_dilatacao, iterations=1)

    removed_lines = np.where(canvas2 > 0.5, 1, preproc)
    return removed_lines

def linhasPorEspacoVert(img):
    pontos = []
    found_border = False
    h, w = img.shape[0], img.shape[1]
    for y in range(1, h):
        last_border = [0,0,0] # Value x-1, value x, x
        curr_border = [0,0,0] # Value x-1, value x, x
        for x in range(1, w):
            curr_pair = [img[y, x-1], img[y, x], x]
            if found_border == True and curr_border [:-1] == [0,0]:
                found_border = False
            if LINE_REPLACE_VALUE in curr_border[:-1]:
                curr_border[:-1] = [0,0]
                found_border = True
            if curr_pair[0] != curr_pair[1]:
                last_border = curr_border
                curr_border = curr_pair
                if curr_border[:-1] == [1,0] and last_border[:-1] == [0,1] and found_border == False:
                    pontos.append([((curr_border[2]-last_border[2])//2)+last_border[2], y])
    return pontos

def linhasPorEspacoHor(img):
    pontos = []
    found_border = False
    h, w = img.shape[0], img.shape[1]
    for x in range(1, w):
        last_border = [0,0,0] # Value y-1, value y, y
        curr_border = [0,0,0] # Value y-1, value y, y
        for y in range(1, h):
            curr_pair = [img[y-1, x], img[y, x], y]
            if found_border == True and curr_border [:-1] == [0,0]:
                found_border = False
            if LINE_REPLACE_VALUE in curr_border[:-1]:
                curr_border[:-1] = [0,0]
                found_border = True
            if curr_pair[0] != curr_pair[1]:
                last_border = curr_border
                curr_border = curr_pair
                if curr_border[:-1] == [1,0] and last_border[:-1] == [0,1] and found_border == False:
                    pontos.append([x, ((curr_border[2]-last_border[2])//2)+last_border[2]])
    return pontos

def regride_linhas_invisiveis(pontos, max_inv_dist):
    pontos.sort()
    i_ant = pontos[0]
    coord_prox = [pontos[0]]
    coord_out = []
    for i in range(1, len(pontos)):
        if pontos[i] - i_ant < max_inv_dist:
            coord_prox.append(pontos[i])
        else:
            if len(coord_prox) >= MIN_SUPPORT_INV_LINE:
                coord_out.append(int(np.round(np.mean(coord_prox).tolist())))
                coord_prox = [pontos[i]]
        i_ant = pontos[i]
    if len(coord_prox) >= MIN_SUPPORT_INV_LINE:
        coord_out.append(np.round(int(np.mean(coord_prox).tolist())))
    return coord_out

def encontra_linhas_invisiveis(blob_img, draw_img, height, width):
    det_inv_lines = np.zeros((height, width), dtype=np.uint8)

    # Verticais
    pontos_linhas_vert = linhasPorEspacoVert(blob_img)
    x_coord = np.array(pontos_linhas_vert)[:,0].tolist()
    linhas_inv_vert = regride_linhas_invisiveis(x_coord, MAX_X_DIST)
    for p in pontos_linhas_vert:
        cv2.circle(draw_img, p, radius=0, color=(0, 255, 0), thickness=-1)
    for x in linhas_inv_vert:
        cv2.line(draw_img,(x,0),(x, height),(0,0,255),2)
        cv2.line(det_inv_lines,(x,0),(x, height),255,2)

    # Horizontais
    pontos_linhas_hor = linhasPorEspacoHor(blob_img)
    y_coord = np.array(pontos_linhas_hor)[:,1].tolist()
    linhas_inv_hor = regride_linhas_invisiveis(y_coord, MAX_Y_DIST)
    for p in pontos_linhas_hor:
        cv2.circle(draw_img, p, radius=0, color=(0, 255, 0), thickness=-1)
    for y in linhas_inv_hor:
        cv2.line(draw_img,(0, y),(width, y),(0,0,255),2)
        cv2.line(det_inv_lines,(0, y),(width, y),255,2)

    det_inv_lines = np.where(det_inv_lines >= 255, 1, 0)

    return det_inv_lines, draw_img

def main ():
    # Abre a imagem em escala de cinza.
    img_or = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img_or is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()
    height, width = img_or.shape[0], img_or.shape[1]
    draw_img = img_or.copy()
    img_or = cv2.cvtColor(img_or, cv2.COLOR_BGR2GRAY)

    preproc = prepocessamento(img_or)
    cv2.imwrite ('_1-preprocessada.png', preproc*255)

    # Canny
    canny = (1-preproc).astype(np.uint8)*255
    canny = cv2.Canny(canny, 50, 150, apertureSize=3)
    cv2.imwrite ('_2-canny.png', canny)

    # Encontra inhas desenhadas
    det_lines, draw_img = encontra_linhas_desenhadas(canny, draw_img, width, height)
    cv2.imwrite ('_3-linhas_detectadas.png', det_lines)

    # Gera a imagem de blobs
    removed_lines = remove_linhas_desenhadas(canny, preproc, height, width)
    kernel_dilatacao = np.ones(KERNEL_DILATACAO_VERTICAL, np.uint8) 
    removed_lines = cv2.dilate(removed_lines, kernel_dilatacao, iterations=1)
    kernel_erosao = np.ones(KERNEL_EROSAO_HORIZONTAL, np.uint8) 
    blob_img = cv2.erode(removed_lines, kernel_erosao, iterations=1)

    cv2.imwrite ('_4-linhas_removidas.png', blob_img*255)

    blob_img = np.where(det_lines > 0.5, LINE_REPLACE_VALUE, blob_img)
    
    # Encontra linhas invisiveis
    det_inv_lines, draw_img = encontra_linhas_invisiveis(blob_img, draw_img, height, width)

    cv2.imwrite ('_5-linhas_invisiveis_detectadas.png', det_inv_lines*255)

    # Combina todas as linhas detectadas
    all_lines = det_lines | det_inv_lines
    cv2.imwrite ('_6_linhas_totais_detectadas.png', all_lines*255)

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
    #     cv2.rectangle(draw_img, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,255))
    # tabela.append(row)

    # Transforma em csv
    filename = INPUT_IMAGE[:-4] + ".csv"
    # with open(filename, 'w') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerows(tabela)

    # c = componentes[83]
    # cv2.imwrite("crop.png", crop)

    cv2.imwrite ('_6-componentes.png', draw_img)
    
    cv2.waitKey ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main ()

#===============================================================================