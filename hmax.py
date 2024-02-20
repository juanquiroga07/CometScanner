#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from collections import deque


# In[2]:


class FastHybridDome:
    def __init__(self, dome_height=20):
        self.dome_height = dome_height

    def setup(self):
        # Puedes implementar la configuración si es necesario
        pass

    def run(self, img):
        height, width = img.shape
        size = height * width

        original_pixels = img.flatten().astype(np.int32)
        dark_pixels = original_pixels - self.dome_height

        # Reconstrucción de la imagen
        for y in range(height):
            for x in range(width):
                max_value = 0
                for i in range(5):
                    X, Y = x, y
                    if i == 0:  # (x+1, y-1)
                        X, Y = x + 1, y - 1
                    elif i == 1:  # (x, y-1)
                        X -= 1
                    elif i == 2:  # (x-1, y-1)
                        X -= 1
                    elif i == 3:  # (x-1, y)
                        Y += 1
                    elif i == 4:  # (x, y)
                        X += 1

                    if 0 <= X < width and 0 <= Y < height:
                        q = dark_pixels[Y * width + X]
                        max_value = max(max_value, q)

                b = y * width + x
                dark_pixels[b] = min(max_value, original_pixels[b])

        for y in range(height - 1, -1, -1):
            for x in range(width - 1, -1, -1):
                max_value = 0
                for i in range(5):
                    X, Y = x, y
                    if i == 0:  # (x, y)
                        pass
                    elif i == 1:  # (x+1, y)
                        X += 1
                    elif i == 2:  # (x+1, y+1)
                        Y += 1
                    elif i == 3:  # (x, y+1)
                        X -= 1
                    elif i == 4:  # (x-1, y+1)
                        X -= 1

                    if 0 <= X < width and 0 <= Y < height:
                        q = dark_pixels[Y * width + X]
                        max_value = max(max_value, q)

                b = y * width + x
                dark_pixels[b] = min(max_value, original_pixels[b])

                for i in range(1, 5):
                    X, Y = x, y
                    if i == 1:  # (x+1, y)
                        X += 1
                    elif i == 2:  # (x+1, y+1)
                        Y += 1
                    elif i == 3:  # (x, y+1)
                        X -= 1
                    elif i == 4:  # (x-1, y+1)
                        X -= 1

                    if 0 <= X < width and 0 <= Y < height:
                        a = Y * width + X  # Asignar 'a' aquí
                        c = Y * width + X
                        q = dark_pixels[c]
                        if q < dark_pixels[b] and q < original_pixels[c]:
                            dark_pixels[c] = min(dark_pixels[a], original_pixels[c])

        fifo = deque()
        for y in range(height):
            for x in range(width):
                a = y * width + x
                for Y in range(y - 1, y + 2):
                    if 0 <= Y < height:
                        offset = Y * width
                        for X in range(x - 1, x + 2):
                            if 0 <= X < width:
                                c = offset + X
                                if dark_pixels[c] < dark_pixels[a] and dark_pixels[c] != original_pixels[c]:
                                    dark_pixels[c] = min(dark_pixels[a], original_pixels[c])
                                    fifo.append((X, Y))

        while fifo:
            x, y = fifo.popleft()
            a = y * width + x  # Mover la definición de 'a' aquí
            for Y in range(y - 1, y + 2):
                if 0 <= Y < height:
                    offset = Y * width
                    for X in range(x - 1, x + 2):
                        if 0 <= X < width:
                            c = offset + X
                            if dark_pixels[c] < dark_pixels[a] and dark_pixels[c] != original_pixels[c]:
                                dark_pixels[c] = min(dark_pixels[a], original_pixels[c])
                                fifo.append((X, Y))

        # Obtiene la cúpula
        result_pixels = original_pixels - dark_pixels
        result_pixels = np.clip(result_pixels, 0, 255).astype(np.uint8)
        result_image = result_pixels.reshape((height, width))

        return result_image


# In[4]:


if __name__ == "__main__":
    # Lee la imagen de entrada
    input_image = cv2.imread("/home/omicas/Documentos/Juanes/Entrenamiento/H-max/Cometas (4)RGB.png", cv2.IMREAD_GRAYSCALE)

    if input_image is None:
        print("Error al cargar la imagen. Asegúrate de que la ruta de la imagen sea correcta.")
    else:
        # Crea una instancia del plugin y ejecútalo
        plugin = FastHybridDome(dome_height=176)
        output_image = plugin.run(input_image)

        # Guarda la imagen de salida en una carpeta
        output_path = "/home/omicas/Documentos/Juanes/Entrenamiento/H-max/resultado2.png"  # Reemplaza con la ruta deseada y el nombre de archivo
        cv2.imwrite(output_path, output_image)
        print(f"Imagen de salida guardada en: {output_path}")


