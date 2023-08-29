# Probelmas que tivemos ::: 

# 1.1 Coloque o camiho da sua máquina no image_path , caso o caminho relativo não funcione (seu aquivo py, ou ipynb tem que estar dentro da pasta (data-retina) para funcionar o caminho relativo )  . Logo Execute Tudo . 

# 1.1* - Caso de um erro do Numpy_... no 1 bloco de código significa que o numpy ou opencv estão desatualizados --> No prompt do Anaconda rode 
# pip install --upgrade opencv-python
# pip install --upgrade numpy 
# :: Caso este erro apareça no seu . No meu funcionou (Arthur)

# Se não solucionar , o (Thales) trouxe outra alternativa que foi criar o Ambiente Virtual (Links no Discord [2 vídeos que ele mandou do yt lá]) No dele funcionou , no meu passei pelo caminho relativo .

# 1.1* este print(image) do 1 Bloco não pode vim como None ou null 

# Caso tudo de certo vai aparecer a imagem , daí é sucesso . / Arthur - Thales

# Fizemos esta parte com a Aula 5 min -1:33:10
# pdf usado :  Modulo 3 - unidade 2 - Pré-processamento de imagenss
# <------------------------------------------------------------- Arthur - Thales (Processamento de Imagens) ------------------------------------------------------------>


# 1 Bloco 

# https://edu.atlanticoavanti.com.br/portal/curso-aula/produto/48482f8bdb4345ccb4972fd96bbf84ea/basico-em-machine-learning

import numpy as np
import cv2 
from matplotlib import pyplot as plt

# 2 jeitos: 
# 1- caminho absoluto ( *** Colocar Caminho da sua Máquina *** ) : 
# image_path = r'C:\Users\01234\OneDrive\Área de Trabalho\Avanti Equipe3\visao-computacional-atlantico\data-retina\test\image\0.png'
# 2- caminho relativo :
image_path = './data-retina/test/image/0.png' 
image= cv2.imread(image_path)

# Resolvi o meu (Arthur) deste jeito: 
# image= cv2.imread('./test/image/0.png')

print(image)

# image_rgb = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR) # isso tava dando erro ao rodar, então modifiquei pra linha 42 - MP
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.axis('off')
plt.show()

if image is not None:
    height, width, channels = image.shape
    for i in range(3):
        pixel = image[i, i]
        print(f'Pixel {i+1}: Valor BGR: {pixel}')
    properties = [
        ('Altura', image.shape[0]),
('Largura', image.shape[1]),
('Canais de cor', image.shape[2]),
('Tipo de dado', image.dtype),
('Valor máximo', image.max()),
('Valor mínimo', image.min()),
('Média', image.mean()),
('Desvio padrão', image.std())
    ]
    for prop_name, prop_value in properties:
        print(f"{prop_name}: {prop_value}")
    cv2.imshow('lmagem - OpenCV', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('Erro ao carregar a imagem.')


# Dividir a imagem em canais de cores
b, g, r = cv2.split(image)
# Exibir os canais de cores usando Matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(131) # Canal R (Vermelho)
plt.imshow(r, cmap='gray')
plt.title('Canal R')
plt.subplot(132) # Canal G (Verde)
plt.imshow(g, cmap='gray')
plt.title('Canal G')
plt.subplot(133) # Canal B (Azul)
plt.imshow(b, cmap='gray')
plt.title('Canal B')
plt.tight_layout()
plt.show()

# Definir o novo tamanho desejado //// Opcional , se for preciso 
new_width = 300
new_height = 200

# Redimensionar a imagem usando o OpenCV
resized_image = cv2.resize(image, (new_width, new_height))
# Mostrar a imagem original
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('lmagem Original')
# Mostrar a imagem redimensionada
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.title('lmagem Redimensionada')
plt.show()

# Converter a imagem para escala de cinza (opcional)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Normalizar a imagem para o intervalo [0, 1]
normalized_image = gray_image / 255.0
print("max tom de cinza:", normalized_image.max())
print("mean tom de cinza:", normalized_image.mean())
print("min tom de cinza:", normalized_image.min())
# Mostrar a imagem gray
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('lmagem gray')

# Mostrar a imagem normalizada
plt.subplot(1, 2, 2)
plt.imshow(normalized_image)
plt.title('lmagem Normalizada')
plt.show()

#  CONTINUAR AQUI DUPLA DE DOMINGO : 

# Numpy necessitou ser importado! Linha 25
# Criadas funções para dar realce no contraste, Equalizar e Obter ranges minimos e máximos

# Função para realce no contraste da imagem
def realce_contraste(imagem, alpha, beta):
    # Aplicar a transformação linear
    imagem_realce = cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)

    return imagem_realce

# Alpha > 1 & Alpha < 3. Beta >= 1 && Beta < 2 *Valores aproximados
imagem_realce = realce_contraste(image, alpha=2, beta=1)

def filtro(image):
    #Aplicar um filtro que torna visível os vasos sanguíneos:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Separando as cores
    r, g, b = cv2.split(image_rgb)
    #Equalizando
    eq_g_canal = cv2.equalizeHist(g)

    return r, g, b, eq_g_canal, image_rgb


def encontrar_range(imagem):
    # Converter a imagem para HSV
    imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # limites inferiores e superiores
    limite_inferior = np.array([12, 50, 50])
    limite_superior = np.array([30, 184, 230])

    imagem_auxiliar = cv2.inRange(imagem_hsv, limite_inferior, limite_superior)

    # Encontre os valores mínimos e máximos nos canais H, S e V para o intervalo de cores
    h_min, s_min, v_min = np.min(imagem_hsv[imagem_auxiliar > 0], axis=0)
    h_max, s_max, v_max = np.max(imagem_hsv[imagem_auxiliar > 0], axis=0)

    return (h_min, s_min, v_min), (h_max, s_max, v_max), imagem_auxiliar

imspecs_min, imspecs_max, imagem_auxiliar = encontrar_range(image)

#Especificações do range de cores da imagem:
print(imspecs_min)
print(imspecs_max)

#Mostrando o efeito do filtro:
im_r, im_g, im_b, eq_g_image, image_rgb = filtro(image)

#Exibindo os resultados
plt.subplot(131)
plt.imshow(image_rgb)
plt.title('Imagem Original')
plt.subplot(132)
plt.imshow(im_g, cmap='gray')
plt.title('Canal G')
plt.subplot(133)
plt.imshow(eq_g_image, cmap='gray')
plt.title('Canal G depois da equalização de histograma')
plt.show()