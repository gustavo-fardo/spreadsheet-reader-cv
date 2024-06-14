
import cv2
import page_skew_corrector
input_image = 'im3.jpg'

img = cv2.imread(input_image,cv2.IMREAD_COLOR)
out,radon,bin = page_skew_corrector.skew_corrector(img)

cv2.imwrite("out.png",out)
cv2.imwrite("radon.png",radon)
cv2.imwrite("bin.png",bin)