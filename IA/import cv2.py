import cv2
import numpy as np
import matplotlib.pyplot as plt
# Leer la imagen en escala de grises
imagen = cv2.imread(r'C:\Users\Usuario\Desktop\IA\Imagen.jpg', cv2.IMREAD_GRAYSCALE)

# Verificación de lectura
if imagen is None:
    print("No se pudo leer la imagen. Verifica la ruta.")
    exit()
# Kernel de realce
kernel = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
])

# Aplicar convolución
resultado = cv2.filter2D(imagen, -1, kernel)
# Mostrar resultados
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(imagen, cmap='gray')
plt.title('Imagen original')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(resultado, cmap='gray')
plt.title('Imagen filtrada (realce)')
plt.axis('off')
plt.show()