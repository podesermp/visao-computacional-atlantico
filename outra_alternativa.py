import numpy as np
import cv2 
from matplotlib import pyplot as plt

# pré processamento
# Contraste: 100,  Temperatura: 0, Matriz: 0, Saturação: 0 e Nitidez: 100


# Carregando a imagem usando o OpenCV
imagem_path = './data-retina/test/image/0.png'
imagem = cv2.imread(imagem_path)
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# Ajustando o contraste
fator_contraste = 2.0
imagem_contraste = cv2.convertScaleAbs(imagem, alpha=fator_contraste, beta=0)

# Convertendo a imagem de BGR para HSV
imagem_hsv = cv2.cvtColor(imagem_contraste, cv2.COLOR_BGR2HSV)

# Ajustando temperatura, matriz e saturação
temperatura = 0
matriz = 0
saturacao = 0
imagem_hsv[:, :, 0] = np.clip(imagem_hsv[:, :, 0] + temperatura, 0, 179)  # Matiz (H)
imagem_hsv[:, :, 1] = np.clip(imagem_hsv[:, :, 1] + matriz, -255, 255)      # Saturação (S)
imagem_hsv[:, :, 2] = np.clip(imagem_hsv[:, :, 2] + saturacao, -255, 255)   # Valor (V)

# Ajustando nitidez
fator_nitidez = 2.0
filtro_nitidez = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
imagem_nitidez = cv2.filter2D(imagem_hsv, -1, filtro_nitidez)


# Convertendo de volta para BGR
imagem_alterada = cv2.cvtColor(imagem_nitidez, cv2.COLOR_HSV2RGB)
imagem_alterada = cv2.GaussianBlur(imagem_alterada, (7,7),0)


# Exibindo a imagem original e a imagem após os pré-processamentos
#Exibindo os resultados
plt.subplot(121)
plt.imshow(imagem_rgb)
plt.title('Imagem RGB')
plt.axis('off')

plt.subplot(122)
plt.imshow(imagem_alterada)
plt.title('Pre-processado')
plt.axis('off')

plt.show()