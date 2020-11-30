from PIL import Image
import numpy as np
from scipy import fft
from scipy import stats
import matplotlib.pyplot as plt
import time


def fuente_info(imagen):
    '''Una función que simula una fuente de
    información al importar una imagen y 
    retornar un vector de NumPy con las 
    dimensiones de la imagen, incluidos los
    canales RGB: alto x largo x 3 canales

    :param imagen: Una imagen en formato JPG
    :return: un vector de pixeles
    '''
    img = Image.open(imagen)
    
    return np.array(img)


def rgb_a_bit(imagen):
    '''Convierte los pixeles de base 
    decimal (de 0 a 255) a binaria 
    (de 00000000 a 11111111).

    :param imagen: array de una imagen 
    :return: Un vector de (1 x k) bits 'int'
    '''
    # Obtener las dimensiones de la imagen
    x, y, z = imagen.shape
    
    # Número total de pixeles
    n_pixeles = x * y * z

    # Convertir la imagen a un vector unidimensional de n_pixeles
    pixeles = np.reshape(imagen, n_pixeles)

    # Convertir los canales a base 2
    bits = [format(pixel,'08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))
    
    return bits_Rx.astype(int)

def modulador(bits, fc, mpp):
    '''Un método que simula el esquema de 
    modulación digital BPSK.

    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La onda portadora c(t)
    :return: La onda cuadrada moduladora (información)
    '''
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits) # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)
    portadora_I = np.sin(2*np.pi*fc*t_periodo)
    portadora_Q = np.cos(2*np.pi*fc*t_periodo)
    
    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp) 
    senal_Tx = np.zeros(t_simulacion.shape)
    moduladora = np.zeros(t_simulacion.shape)  # señal de información
    primer_bit = [] 
    segundo_bit = []
    
    # 3.1 Crea dos vectores de bits separados
    
    for i, aux in enumerate(bits):
        if i%2 == 0:
            primer_bit.append(aux)
        else: segundo_bit.append(aux)
        
   
    # 4. Asignar las formas de onda según los bits (BPSK)
    for i, bit in enumerate(primer_bit):
        if bit == 1:
            senal_Tx[2*i*mpp : 2*i*mpp + mpp] = portadora_I
            moduladora[2*i*mpp : 2*i*mpp + mpp] = 1
           
        else:
            senal_Tx[2*i*mpp : 2*i*mpp + mpp] = portadora_I* -1
            moduladora[2*i*mpp : 2*i*mpp + mpp] = 0
                       
    for i, bit in enumerate(segundo_bit):
        if bit == 1:
            senal_Tx[2*(i*mpp) +1*mpp : 2*((i+1)*mpp)] = portadora_Q
            moduladora[2*(i*mpp) +1*mpp : 2*((i+1)*mpp)] = 1
     
        else:
            senal_Tx[2*(i*mpp)+ 1*mpp : 2*((i+1)*mpp)] = portadora_Q * -1
            moduladora[2*(i*mpp) +1*mpp : 2*((i+1)*mpp)] = 0
            
    
    # 5. Calcular la potencia promedio de la señal modulada
    Pm = (1 / (N*Tc)) * np.trapz(np.power(senal_Tx, 2), t_simulacion)
     
    return senal_Tx, Pm, portadora_I, portadora_Q, moduladora


def canal_ruidoso(senal_Tx, Pm, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.

    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido
 
    return senal_Rx

def demodulador(senal_Rx, portadora_I, portadora_Q, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema BPSK. El criterio
    de demodulación se basa en decodificación por 
    detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora: La onda portadora c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)
 

    # Cantidad de bits en transmisión
    N = int(M / mpp)
    
    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(M)

    
    # Demodulación
    for i in range(N//2):
        # Producto interno de dos funciones
        producto_I = senal_Rx[2*i*mpp : 2*i*mpp + mpp] * portadora_I
        senal_demodulada[2*i*mpp : 2*i*mpp + mpp] = producto_I
        Ep_I = np.sum(producto_I) 

        # Criterio de decisión por detección de energía
        if Ep_I > 0:
            bits_Rx[2*i] = 1
        else:
            bits_Rx[2*i] = 0
            
    for i in range(N//2):
        # Producto interno de dos funciones
        producto_Q = senal_Rx[2*(i*mpp) +1*mpp : 2*((i+1)*mpp)] * portadora_Q
        senal_demodulada[2*(i*mpp) +1*mpp : 2*((i+1)*mpp)] = producto_Q
        Ep_Q = np.sum(producto_Q) 

        # Criterio de decisión por detección de energía
        if Ep_Q > 0:
            bits_Rx[2*i +1] = 1
        else:
            bits_Rx[2*i +1] = 0


    return bits_Rx.astype(int), senal_demodulada

def bits_a_rgb(bits_Rx, dimensiones):
    '''Un blque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k 
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)


# Parámetros
fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = -5    # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)

# 3. Modular la cadena de bits usando el esquema BPSK
senal_Tx, Pm, portadora_I, portadora_Q, moduladora= modulador(bits_Tx, fc, mpp)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx, senal_demodulada = demodulador(senal_Rx, portadora_I, portadora_Q, mpp)

# 6. Se visualiza la imagen recibida 
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recuperado')
Fig.tight_layout()

plt.imshow(imagen_Rx)


# Visualizar el cambio entre las señales
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(14, 7))

# La onda cuadrada moduladora (bits de entrada)
ax1.plot(moduladora[0:600], color='r', lw=2) 
ax1.set_ylabel('$b(t)$')

# La señal modulada por BPSK
ax2.plot(senal_Tx[0:600], color='g', lw=2) 
ax2.set_ylabel('$s(t)$')

# La señal modulada al dejar el canal
ax3.plot(senal_Rx[0:600], color='b', lw=2) 
ax3.set_ylabel('$s(t) + n(t)$')

# La señal demodulada
ax4.plot(senal_demodulada[0:600], color='m', lw=2) 
ax4.set_ylabel('$b^{\prime}(t)$')
ax4.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()

#----------------- Parte 2--------------
# Creación del vector de tiempo y frecuencia
creator = []
N = 9
t_simulacion = np.linspace(0, N*Tc, N*mpp) 
Tc = 1 / fc  # periodo [s]

P = [np.mean(senal_Tx[i]) for i in range(len(t_simulacion))]

plt.plot(t_simulacion, P, lw=4)
plt.title('Realizaciones del proceso aleatorio $X(t)$')
plt.xlabel('$t$')
plt.ylabel('$x_i(t)$')
plt.show()


#-----------------Parte 3---------------


# Transformada de Fourier
senal_f = fft(senal_Tx)

# Muestras de la señal
Nm = len(senal_Tx)

# NUmero de simbolos 
Ns = Nm // mpp

# Tiempo del simbolo

Tc = 1 / fc
 
# Tiempo entre muestras
Tm = Tc / mpp

#TIempo de la simulacion
T = Ns * Tc

# Espacio de frecuencias
f = np.linspace(0.0, 1.0/(2.0*Tm), Nm//2)

#Grafica
plt.plot(f, 2.0/Nm * np.power(np.abs(senal_f[0:Nm//2]),2))
plt.xlim(0,20000)
plt.grid()
plt.show()
