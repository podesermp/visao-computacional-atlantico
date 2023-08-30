import numpy as np
import cv2 
from matplotlib import pyplot as plt


# Mostrar uma imagem
def show(img, title=""):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Aplicar clahe em uma imagem (1-D)
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    clahe_image = clahe.apply(img)
    return clahe_image

# Aplicar filtro mediana em uma imagem
def filter_med(img):
    imgMed = cv2.medianBlur(img, 5)
    return imgMed

# Aplicar filtro gaussiano em uma imagem
def filter_gauss(img):
    imgGauss = cv2.GaussianBlur(img, (3,3), 0)
    return imgGauss

image_path = './data-retina/test/image/0.png' 
image= cv2.imread(image_path)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

r,g,b = cv2.split(image_rgb)

r = apply_clahe(r)
g = apply_clahe(g)
b = apply_clahe(b)

img_clahe = cv2.merge([r,g,b])
img_med = filter_med(img_clahe)
img_preprocessed = filter_gauss(img_med)


#Exibindo os resultados
plt.subplot(331)
plt.imshow(r)
plt.axis('off')
plt.title('Canal Red')

plt.subplot(332)
plt.imshow(g)
plt.axis('off')
plt.title('Canal Green')

plt.subplot(333)
plt.imshow(b)
plt.title('Canal Blue')
plt.axis('off')

plt.subplot(334)
plt.imshow(image_rgb)
plt.title('Imagem original')
plt.axis('off')

plt.subplot(335)
plt.imshow(img_clahe)
plt.title('Imagem pós Clahe')
plt.axis('off')

plt.subplot(336)
plt.imshow(img_med)
plt.title('Imagem filtrada (medianBlur)')
plt.axis('off')

plt.subplot(338)
plt.imshow(img_preprocessed)
plt.title('Imagem filtrada (GaussianBlur) - FINAL')
plt.axis('off')
plt.show()