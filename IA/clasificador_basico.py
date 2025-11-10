import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- CARGA DE LAS IMÁGENES ---
# Se pueden reemplazar las rutas por las de tus imágenes locales
imagen1 = cv2.imread("C:/Users/Usuario/Desktop/IA/imagen1.jpg")  # Imagen RGB
imagen2 = cv2.imread("C:/Users/Usuario/Desktop/IA/imagen2.jpg")
imagen3 = cv2.imread("C:/Users/Usuario/Desktop/IA/imagen3.jpg")

# Verificar que las imágenes se hayan leído correctamente
if imagen1 is None or imagen2 is None or imagen3 is None:
    print(" Error al cargar las imágenes. Verifica las rutas.")
    exit()

# Convertir de BGR (formato de OpenCV) a RGB para mostrar correctamente con matplotlib
imagen1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2RGB)
imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2RGB)
imagen3 = cv2.cvtColor(imagen3, cv2.COLOR_BGR2RGB)

# --- CONVERSIÓN A ESCALA DE GRISES ---
gris1 = cv2.cvtColor(imagen1, cv2.COLOR_RGB2GRAY)
gris2 = cv2.cvtColor(imagen2, cv2.COLOR_RGB2GRAY)
gris3 = cv2.cvtColor(imagen3, cv2.COLOR_RGB2GRAY)

# --- CONVERSIÓN A BLANCO Y NEGRO (BINARIZACIÓN) ---
# Se utiliza un umbral de 127 para separar tonos claros y oscuros
_, bn1 = cv2.threshold(gris1, 127, 255, cv2.THRESH_BINARY)
_, bn2 = cv2.threshold(gris2, 127, 255, cv2.THRESH_BINARY)
_, bn3 = cv2.threshold(gris3, 127, 255, cv2.THRESH_BINARY)

# --- CONVERSIÓN DE BLANCO Y NEGRO A RGB ---
rgb_bn1 = cv2.cvtColor(bn1, cv2.COLOR_GRAY2RGB)
rgb_bn2 = cv2.cvtColor(bn2, cv2.COLOR_GRAY2RGB)
rgb_bn3 = cv2.cvtColor(bn3, cv2.COLOR_GRAY2RGB)

# --- VISUALIZACIÓN DE LOS RESULTADOS ---
fig, axs = plt.subplots(3, 4, figsize=(12, 8))
fig.suptitle("Clasificación y Conversión de Imágenes", fontsize=14)

imagenes = [
    (imagen1, gris1, bn1, rgb_bn1),
    (imagen2, gris2, bn2, rgb_bn2),
    (imagen3, gris3, bn3, rgb_bn3)
]

titulos = ["RGB original", "Escala de grises", "Blanco y negro", "RGB reconstruido"]

for i in range(3):
    for j in range(4):
        axs[i, j].imshow(imagenes[i][j], cmap='gray' if j > 0 else None)
        axs[i, j].set_title(titulos[j])
        axs[i, j].axis('off')

plt.tight_layout()
plt.show()

# --- CLASIFICACIÓN SEGÚN BRILLO ---
# Cálculo del brillo promedio de cada imagen
def calcular_brilho_promedio(img):
    if len(img.shape) == 3:  # Imagen RGB
        return np.mean(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    else:  # Imagen en escala de grises
        return np.mean(img)

brillos = [
    calcular_brilho_promedio(imagen1),
    calcular_brilho_promedio(imagen2),
    calcular_brilho_promedio(imagen3)
]

print("\n CLASIFICACIÓN SEGÚN BRILLO PROMEDIO:")
for i, brillo in enumerate(brillos, 1):
    clasificacion = "Brillante" if brillo > 127 else "Oscura"
    print(f"Imagen {i}: {clasificacion} (Brillo promedio = {brillo:.2f})")
    print("\n Proceso completado con éxito.")

