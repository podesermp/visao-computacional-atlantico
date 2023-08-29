import numpy as np
import cv2 
from matplotlib import pyplot as plt


# Mostrar uma imagem
def show(img, title=""):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Aplicar a transformação linear para aumentar o contraste da imagem
def realce_contraste(imagem, alpha, beta):
    imagem_realce = cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)

    return imagem_realce

# Aplicar um filtro que torna visível os vasos sanguíneos:
def filtro(image):
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # comentei isso pq ja recebo como rgb
    
    # Separando as cores
    r, g, b = cv2.split(image)
    
    # Equalizando
    g = cv2.equalizeHist(g)
    b = cv2.equalizeHist(b)

    #juntando os canais
    imagem_equalizada = cv2.merge([r,g,b])
    return r, g, b, imagem_equalizada

image_path = './data-retina/test/image/0.png' 
image= cv2.imread(image_path)


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Alpha > 1 & Alpha < 3. Beta >= 1 && Beta < 2 *Valores aproximados
imagem_realcada = realce_contraste(image_rgb, alpha=2, beta=1)

r, g, b, imagem_preprocessada = filtro(imagem_realcada)

# show(imagem_realcada, 'Imagem realçada')

#Exibindo os resultados
plt.subplot(231)
plt.imshow(r)
plt.axis('off')
plt.title('Canal Red')

plt.subplot(232)
plt.imshow(g)
plt.axis('off')
plt.title('Canal Green')

plt.subplot(233)
plt.imshow(b)
plt.title('Canal Blue')
plt.axis('off')

plt.subplot(234)
plt.imshow(image_rgb)
plt.title('Imagem original')
plt.axis('off')

plt.subplot(235)
plt.imshow(imagem_realcada)
plt.title('Imagem realçada')
plt.axis('off')

plt.subplot(236)
plt.imshow(imagem_preprocessada, cmap='gray')
plt.title('Imagem pré processada')
plt.axis('off')
plt.show()