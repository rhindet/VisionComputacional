import cv2
import numpy as np

def guardarImagenMatrizBGR(img):
    # Guardar imagen de matriz de imagen en BGR
    #img es un array NumPy de 3 dimensiones:
	#•	Eje 0 → alto (filas de píxeles)
	#•	Eje 1 → ancho (columnas de píxeles)
	#•	Eje 2 → canales de color (B, G, R)
    #print(img.shape) ->  (alto, ancho, 3)
    #reshape cambia la forma de la matriz sin cambiar sus valores.
    #-1 le dice  a NumPy: “calcula automáticamente cuántas filas necesito”.
    # img.shape[2] es el número de canales de color(normalmente 3 → BGR).
    # Si la imagen era 100 × 200 × 3, el reshape la transforma en 20000 × 3.
    # Cada fila representa un píxel y las tres columnas son B, G, R.
    #np.savetxt escribe en texto plano.
    #fmt='%d' significa escribe como enteros (sin decimales).
	#Como los valores de cada canal de color van de 0 a 255, es correcto usar enteros.
    try:
        cv2.imwrite("imagen_bgr.png", img)
        np.savetxt("Imagen_BGR_Matriz.txt", img.reshape(-1, img.shape[2]), fmt='%d')
        print("Se guardo guardarImagenBGR con exito")
    except:
        print("Error al guardar imagen en guardarImagenBGR()")

def convertirImagenMatrizRGB(img):
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("imagen_rgb.png", img_rgb)
        np.savetxt("Imagen_RGB_Matriz.txt", img_rgb.reshape(-1, img_rgb.shape[2]), fmt='%d')
        print("Se guardo guardarImagenMatrizRGB con exito")
        return img_rgb
    except:
        print("Error al guardar imagen en guardarImagenMatrizRGB()")

def convertirAEscalaDeGrises(img):
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("imagen_Grises.png", img_gray)
        np.savetxt("Imagen_EscalaGrises_Matriz.txt", img_gray, fmt='%d')
        print("Se guardo convertirAEscalaDeGrises con exito")
        return  img_gray

    except:
        print("Error al convertir imagen en convertirAEscalaDeGrises()")

def convertirNegativoBGR(img):
    try:
        neg = cv2.bitwise_not(img)

        # Guardar imagen
        cv2.imwrite("imagen_negativa_bgr.png", neg)

        # Guardar matriz en TXT (cada fila = [B, G, R])
        h, w, c = neg.shape
        np.savetxt("imagen_negativa_bgr.txt", neg.reshape(-1, c), fmt='%d')

        print("Se convertirNegativoBGR con éxito")
        return neg

    except Exception as e:
        print("Error al convertir imagen en convertirNegativoBGR():", e)

def convertirNegativoRGB(img):
    try:
        neg = cv2.bitwise_not(img)

        # Guardar imagen
        cv2.imwrite("imagen_negativa_rgb.png", neg)

        # Guardar matriz en TXT (cada fila = [B, G, R])
        h, w, c = neg.shape
        np.savetxt("imagen_negativa_rgb.txt", neg.reshape(-1, c), fmt='%d')

        print("Se convertirNegativoBGR con éxito")
        return neg

    except Exception as e:
        print("Error al convertir imagen en convertirNegativoBGR():", e)

def convertirNegativoEscalaDeGrises(img):
    try:
        neg = cv2.bitwise_not(img)

        # Guardar la imagen
        cv2.imwrite("imagen_negativa_grises.png", neg)

        # Guardar la matriz como texto
        np.savetxt("imagen_negativa_grises.txt", neg, fmt='%d')

        print("Se convirtió a negativo y se guardó matriz con éxito")
    except Exception as e:
        print("Error:", e)

def convertirGrisesConContraste(img,alpha,beta,):
    try:
        out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)  # aplica y recorta al rango
        cv2.imwrite("imagen_contraste_grises.png", out)

        # Guardar la matriz como texto
        np.savetxt("imagen_contraste_grises.txt", out, fmt='%d')
        print("Se convirtió a convertirGrisesConContraste y se guardó matriz con éxito")
        return out

    except Exception as e:
        print("Error:", e)

def convertirBGRConContraste(img,alpha,beta,):
    try:
        out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)  # aplica y recorta al rango
        cv2.imwrite("imagen_contraste_bgr.png", out)

        # Guardar matriz en TXT (cada fila = [B, G, R])
        h, w, c = out.shape
        np.savetxt("imagen_contraste_bgr.txt", out.reshape(-1, c), fmt='%d')

        print("Se convirtió a convertirBGRConContraste y se guardó matriz con éxito")
        return out

    except Exception as e:
        print("Error:", e)


# 1) Copiado (bit a bit)
def copiarImagen(img):
    copia = img.copy()
    cv2.imwrite("imagen_copiada.png", copia)
    return copia

# 2) Aclarado / Oscurecimiento (operación puntual con saturación)
# k > 0 aclara, k < 0 oscurece
def aclarar_oscurecer(img, k=30):
    # img puede ser GRAY (H,W) o BGR (H,W,3)
    out = cv2.add(img, k) if k >= 0 else cv2.subtract(img, -k)
    cv2.imwrite("imagen_aclarada_oscurecida.png", out)
    return out

# (Si quieres “brillo” explícito usando la misma idea:)
def brillo(img, beta=30):
    out = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)  # solo brillo
    cv2.imwrite("imagen_brillo.png", out)
    return out

def separarCanalesRGB(img_rgb):
    # Dividir los tres canales
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    # Guardar cada canal en escala de grises
    cv2.imwrite("canal_R.png", R)
    cv2.imwrite("canal_G.png", G)
    cv2.imwrite("canal_B.png", B)

    # Crear imágenes donde solo un canal está activo en color
    zeros = np.zeros_like(R)

    solo_rojo  = cv2.merge([zeros, zeros, R])   # R visible
    solo_verde = cv2.merge([zeros, G, zeros])   # G visible
    solo_azul  = cv2.merge([B, zeros, zeros])   # B visible

    cv2.imwrite("solo_rojo.png", solo_rojo)
    cv2.imwrite("solo_verde.png", solo_verde)
    cv2.imwrite("solo_azul.png", solo_azul)

    np.savetxt("solo_rojo.txt", solo_rojo.reshape(-1, solo_rojo.shape[2]), fmt='%d')
    np.savetxt("solo_verde.txt", solo_verde.reshape(-1, solo_verde.shape[2]), fmt='%d')
    np.savetxt("solo_azul.txt", solo_azul.reshape(-1, solo_azul.shape[2]), fmt='%d')


def leerImagen():

    #Cargar imagen con openCV (color por defecto: BGR)
    img = cv2.imread("alemania.png",cv2.IMREAD_COLOR)

    #Validar si la imagen es vaalida
    if img is None:
        raise FileNotFoundError("No se pudo cargar la imagen.Revisa la ruta")

    #Guardar imagen en BGR
    guardarImagenMatrizBGR(img)

    #Guardar imagen en RGB (BGR -> RGB)
    img_rgb = convertirImagenMatrizRGB(img)

    #Convertir imagen en Escala de grises (BGR -> GRISES)
    img_grises = convertirAEscalaDeGrises(img)

    #Convertir imagen de BGR a Negativo
    img_negativo_bgr = convertirNegativoBGR(img)

    #Convertir imagen RGB a Negativo
    img_negativo_rgb = convertirNegativoRGB(img_rgb)

    #Convertir imagen de Escala de grises a negativo
    convertirNegativoEscalaDeGrises(img_grises)

    #Moficaar imagen de Escala de grises con contraste (mas negro el negro y mas blanco el blanco)
    #Con alpha=10, prácticamente t0do se satura a blanco, salvo los negros más bajos.
    contraste_girses = convertirGrisesConContraste(img_grises, alpha=10, beta=0)

    # Moficaar imagen de Escala de grises con contraste
    contraste_bgr = convertirBGRConContraste(img, alpha=1, beta=0)

    _ = copiarImagen(img)
   # _ = aclarar_oscurecer(img, k=100)  # aclara
    _ = aclarar_oscurecer(img, k=-100)  # oscurece
    # Solo brillo (en color)
    #_ = brillo(img, beta=40)  # +brillo
    _ = brillo(img, beta=-40)  # -brillo

    separarCanalesRGB(img_rgb)

    #Mostrar imagen con openCV
    cv2.imshow("Imagen",img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    leerImagen()
