
import cv2
import page_skew_corrector
input_image = 'amostras/6.jpg'

img = cv2.imread(input_image,cv2.IMREAD_COLOR)
out,radon,bin = page_skew_corrector.skew_corrector(img, percentage_cut=0.15)

cv2.imwrite("_0.1-corrected_out.png",out)
cv2.imwrite("_0.2-radon.png",radon*255)
cv2.imwrite("_0.3-bin.png",bin*255)