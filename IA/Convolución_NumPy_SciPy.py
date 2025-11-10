import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# --- 1. Leer la imagen en escala de grises ---
imagen = cv2.imread(r'C:/Users/Usuario/Desktop/IA/imagen1.jpg', cv2.IMREAD_GRAYSCALE)

# Verificación de lectura
if imagen is None:
    print("No se pudo leer la imagen. Verifica la ruta.")
    exit()

# --- 2. Filtro de Suavizado (promedio) ---
kernel_suave = np.ones((5,5), np.float32) / 25
img_suavizada = cv2.filter2D(imagen, -1, kernel_suave)

# --- 3. Filtro de Realce (sharpen) ---
kernel_realce = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
img_realzada = cv2.filter2D(imagen, -1, kernel_realce)

# --- 4. Filtro de Bordes (Laplaciano) usando convolución manual ---
kernel_bordes = np.array([[-1, -1, -1],
                          [-1, 8, -1],
                          [-1, -1, -1]])

resultado_bordes = convolve2d(imagen, kernel_bordes, mode='same', boundary='fill', fillvalue=0)

# Normalizar el resultado al rango 0-255
img_bordes = cv2.normalize(resultado_bordes, None, 0, 255, cv2.NORM_MINMAX)
img_bordes = np.uint8(img_bordes)

# --- 5. Guardar resultados ---
cv2.imwrite('suavizada.jpg', img_suavizada)
cv2.imwrite('realzada.jpg', img_realzada)
cv2.imwrite('bordes.jpg', img_bordes)

# --- 6. Mostrar las tres imágenes juntas ---
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img_suavizada, cmap='gray')
plt.title('Suavizada')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(img_realzada, cmap='gray')
plt.title('Realzada')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(img_bordes, cmap='gray')
plt.title('Bordes')
plt.axis('off')

plt.show()
