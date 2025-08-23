# This is a sample Python script.

import cv2
import matplotlib.pyplot as plt
import numpy as np

from cv2 import COLOR_RGB2GRAY
from cv2 import imshow, waitKey
from cv2 import cvtColor, COLOR_BGR2RGB


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    image_path = "dumbbell.png"



    #RGB SCALE
    img = cv2.imread(image_path)
    img_rgb = cvtColor(img, COLOR_BGR2RGB)
    #plt.imshow(img_rgb)
    #print(img_rgb)
    #plt.show()


    img_gray = cvtColor(img_rgb, COLOR_RGB2GRAY)
    imshow('GrayscaleImage', img_gray)
    print(img_gray)
    print(img_gray[0, 0])
    waitKey(0)
    print("Image Dimensions = {}\n".format(img.shape))
    print(img.shape[0])




    return



    print(img[0, 0])
    plt.imshow(img)
    plt.title('Displaying image using Matplotlib')
    plt.show()
    imshow('Displaying image using OpenCV', img)
    waitKey(0)
    #plt.imshow(img)
    #plt.show()

    #cv2.imshow('asd', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    #img = cv2.imread(image_path, 0)
    ##plt.imshow(img,cmap="gist_rainbow")
    #plt.show()
    #print(img[0])


    #print(img)
    #print("Data type = {}\n".format(img.dtype))
    print("Object type = {}\n".format(type(img)))
    print("Image Dimensions = {}\n".format(img.shape))

def paintMatrix():
    # Matriz 50x50 negra
    img = np.zeros((50, 50), dtype=np.uint8)

    # "2" blanco (255) — trazos simples
    img[0:1,0:50] = 100

    img[10:12, 15:35] = 255     # parte superior
    img[12:25, 33:35] = 255     # parte superior derecha
    img[23:25, 15:35] = 255     # parte media
    img[25:40, 15:17] = 255     # parte inferior izquierda
    img[38:40, 15:35] = 255     # parte inferior

    # Mostrar en escala de grises
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Número 2 en matriz 50x50")
    ax.axis("off")

    # Función para manejar el clic del ratón
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            value = img[y, x]   # fila = y, columna = x
            print(f"Pixel ({x}, {y}) -> Valor: {value}")

    # Conectar el evento
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
    return img


def loadImage():
    # Ruta de la imagen
    image_path = "dumbbell.png"

    # Leer con opencv
    img = cv2.imread(image_path)

    if img is None:
        print("Error: No se pudo cargar la imagen, revisar la ruta.")
        return

    # Convertir BGR (OpenCV) → RGB (para matplotlib)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Guardar matriz en archivo .txt
    np.savetxt("matriz_imagen.txt", img_rgb.reshape(-1, 3), fmt="%d")
    print("Matriz guardada en matriz_imagen.txt")

    # Redimensionar a 24x24
    small_rgb = cv2.resize(img_rgb, (24, 24), interpolation=cv2.INTER_AREA)

    # Guardar versión escala de grises
    gray_small = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
    np.savetxt("matriz_24x24.txt", gray_small, fmt="%3d")
    print("Matriz 24x24 en GRIS guardada en matriz_24x24.txt")

    # Guardar versión RGB en binario (mantiene la forma 24x24x3)
    np.save("matriz_24x24_rgb.npy", small_rgb)
    print("Matriz RGB 24x24 guardada en matriz_24x24_rgb.npy")

    # ✅ Guardar versión RGB legible como texto
    with open("matriz_24x24_rgb.txt", "w") as f:
        for y in range(24):
            fila = " ".join(f"{tuple(small_rgb[y, x])}" for x in range(24))  # ej: (R,G,B)
            f.write(fila + "\n")
    print("Matriz RGB 24x24 guardada en matriz_24x24_rgb.txt")

    # Mostrar con Matplotlib
    plt.imshow(img_rgb)
    plt.title("Imagen cargada")
    plt.axis("off")
    plt.show()

    return img_rgb

def loadImage2():
    image_path = "dumbbell.png"
    img = cv2.imread(image_path)
    if img is None:
        print("Error: No se pudo cargar la imagen, revisa la ruta.")
        return

    # --- Reemplazar azul → verde ---
    # Pasar a HSV para segmentar por color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Rango aproximado de "azul" en OpenCV (H: 0-179)
    lower_blue = np.array([90, 80, 50], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Copia y reemplazo: donde hay azul, pon verde (BGR = (0,255,0))
    img_replaced = img.copy()
    img_replaced[mask > 0] = (0, 255, 0)

    # Mostrar original vs modificado
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(img_replaced, cv2.COLOR_BGR2RGB))
    plt.title("Azul → Verde")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return img_replaced

import cv2
import numpy as np

def loadImage3():
    image_path = "imagen_prueba.jpg"
    img = cv2.imread(image_path)

    if img is None:
        print("Error: No se pudo cargar la imagen, revisa la ruta.")
        return None

    # Convertir BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Redimensionar a 24x24
    img_resized = cv2.resize(img_rgb, (24, 24), interpolation=cv2.INTER_AREA)

    # === Copia del 24x24 original y modificación del primer pixel ===
    img_resized_orig = img_resized.copy()

    # Cambia el primer pixel (fila 0, columna 0) a ROJO (R,G,B)
    img_resized[0, 0] = [255, 0, 0]
    img_resized[10, 20] = [0, 0, 255]


    # Guardar matriz MODIFICADA en archivo con formato 24x24 (tuplas por píxel)
    with open("matriz_24x24_rgb_mod.txt", "w") as f:
        for fila in img_resized:
            linea = "  ".join([f"({r:3},{g:3},{b:3})" for r, g, b in fila])
            f.write(linea + "\n")
    print("Matriz 24x24 modificada guardada en 'matriz_24x24_rgb_mod.txt'")

    # Comparar visualmente: original vs modificado
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized_orig)
    plt.title("24x24 original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_resized)
    plt.title("24x24 modificado (0,0)=rojo")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Guardar la matriz en archivo con formato 24x24
    with open("matriz_24x24_rgb.txt", "w") as f:
        for fila in img_resized_orig:
            linea = "  ".join([f"({r:3},{g:3},{b:3})" for r, g, b in fila])
            f.write(linea + "\n")

    print("✅ Matriz 24x24 guardada en 'matriz_24x24_rgb.txt'")

    return img_resized

# Ejecutar
#matriz = loadImage()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #paintMatrix()
    loadImage3()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
