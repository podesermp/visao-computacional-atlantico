import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import copy
import skimage.filters
import skimage.exposure
import skimage.filters.rank
import skimage.morphology
import scipy.ndimage
import os
import matplotlib.image as img
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import hessian
from skimage import morphology
import pywt

link = []
link = glob.glob('D:/*.png')

url = 'D://1.png'
img = cv2.imread(url)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#separar canais
imgR = img[:,:,0]
imgG = img[:,:,1]
imgB = img[:,:,2]

#função de desenho do histograma
def drawHistogram(img):
    his = np.zeros(256,)
    a = img.flatten()
    for i in a:
        his[i] = his[i]+1
    return his

#aplicando clahe no canal verde da imagem
clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
clahe_img = clahe.apply(imgG)
plt.hist(clahe_img.flat, bins=100, range=(0,255))

#Operações morfológicas, realizamos a abertura e, em seguida, o fechamento.
cell_disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
Topen = cv2.morphologyEx(clahe_img,cv2.MORPH_OPEN,cell_disc)
Tclose = cv2.morphologyEx(Topen, cv2.MORPH_CLOSE, cell_disc)
# cell_disc1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# Tclose = cv2.morphologyEx(Topen, cv2.MORPH_DILATE, cell_disc1)

#Tophat imagem
TopHat = (clahe_img - Tclose)#.astype(np.uint8)

#Mais operações morfológicas, erosão seguida de dilatação.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
min_image = cv2.erode(TopHat, kernel)
min_image = cv2.dilate(min_image, kernel)


#thresholding
def threshold(img,k):
    ret = copy.deepcopy(img)
    ret[ret<k] = 0
    ret[ret>=k] = 255
    return ret

#Abordagem da matriz de Hessian e dos autovalores para obter uma imagem de vasos finos aprimorada
HessThin = hessian_matrix(min_image, sigma=1.2, order='rc')
EignThin = hessian_matrix_eigvals(HessThin) [1]
#Abordagem da matriz de Hessian e dos autovalores para obter uma imagem de vasos largos aprimorada.
HessWide = hessian_matrix(min_image, sigma=4, order='rc')
EignWide = hessian_matrix_eigvals(HessWide) [1]

#Otsu GLOBAL thresholding
def GlobalOtsu(img):
    foreground = img[img>=0]
    background = img[img<0]
    
    final_var = (np.var(foreground) * len(foreground) + np.var(background) * len(background))/(len(foreground) + len(background))
    if(np.isnan(final_var)):
        final_var = -1
        
    final_thresh = 0
    for i in np.linspace(np.min(img), np.max(img), num=255):
        foreground = img[img>=i]
        background = img[img<i]
        var = (np.var(foreground) * len(foreground) + np.var(background) * len(background))/(len(foreground) + len(background))
        
        if(np.isnan(var)):
            var = -1
            
        if(var!=-1 and (var<final_var or final_var ==-1)):
            final_var = var
            final_thresh = i
    return threshold(img,final_thresh)

#Limiarização/baseada em área para limpeza. Realizado como etapa de pós-processamento.
def AreaThreshold(img, area = 5):
    nlabels,labels,stats,centroid = cv2.connectedComponentsWithStats(np.uint8(img), 4, cv2.CV_32S)

    output = np.copy(img)
    
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if stats[labels[i][j], cv2.CC_STAT_AREA] < area:
                output[i][j] = 0
                
    return output

#Otsu LOCAL thresholding
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk

#Função de limiarização Otsu local conforme o artigo, SEM inclusão de deslocamento.
def LocalOtsu1(img,radius = 5):
    selem = disk(radius)

    local_otsu = rank.otsu(img, selem)
    output = np.copy(img)
    
    output[output < local_otsu] = 0
    output[output >= local_otsu] = 255
    
    return output
#Função de limiarização Otsu local conforme o artigo, COM inclusão de deslocamento.
def LocalOtsu2(img,radius = 15):
    selem = disk(radius)

    local_otsu = rank.otsu(img, selem)
    output = np.copy(img)
    rng = local_otsu.max() - local_otsu.min()
    mid = rng/2 + local_otsu.min()

    local_otsu[local_otsu<mid] = mid

    output[output < local_otsu] = 0

    return output

#Utilizando o método da transformada wavelet para fusão de imagens.
def image_fusion(img1,img2):
    w1 = pywt.wavedec2(img1, 'db1')
    w2 = pywt.wavedec2(img2, 'db1')
    elem = (w1[0]+w2[0])/2
    fw = [elem]
    
    for i in range(len(w1)-1):
        x,y,z = (w1[i+1][0] + w2[i+1][0])/2, (w1[i+1][1] + w2[i+1][1])/2, (w1[i+1][2] + w2[i+1][2])/2
        fw.append((x,y,z))

    output = pywt.waverec2(fw, 'db1')
    
#Normalização, pode ou não ser necessária.
    amin = np.min(output)
    amax = np.max(output)
    output = 255* ((output - amin)/(amax-amin))
    
    output = cv2.resize(output,img1.T.shape)
    return output

#Aplicando o Otsu GLOBAL
val1 = GlobalOtsu(1-EignWide)

#Normalizando imagens
#Duas imagens que precisam ser fundidas.
thinN = cv2.normalize(1-EignThin,  None, 0, 255, cv2.NORM_MINMAX)
val1 = cv2.normalize(val1,  None, 0, 70, cv2.NORM_MINMAX)

#Fundindo as imagens
test1 = image_fusion(val1,thinN)

#Aplicando Otsu LOCAL
lOtsu = LocalOtsu2(test1.astype('uint8'))

#Limiarização de área para remover regiões que não são de vasos.
final = AreaThreshold(lOtsu,50)

#Binariando a saída final e visualizando-a.
final[final!=0] = 255

#Recebendo a mascara segmentada do dataset
imgRef = cv2.imread("C:/Users/Francisco/Desktop/Retina/data-retina/train/mask/1.png")
imgRef = imgRef[:,:,1]

cv2.imshow("Imagem Final", final)
cv2.waitKey()
cv2.destroyAllWindows()

#Funções de métricas do Renê
import function as f

print(f.compute_dice_similarity(final, imgRef))
print(f.compute_fit_adjust(final, imgRef))
print(f.compute_position_adjust(final, imgRef))
print(f.compute_size_adjust(final, imgRef))

