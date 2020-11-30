---

## Universidad de Costa Rica
### Escuela de Ingeniería Eléctrica
#### IE0405 - Modelos Probabilísticos de Señales y Sistemas

Segundo semestre del 2020

---

* Estudiante: **Jorge Daniel Rodríguez Hernández**
* Carné: **A95284**
* Grupo: **2**

# `P4` - *Modulación digital IQ*

Para obtener una modulación digital de una señal con un ruido ambiental simulado se decidio utilizar el codigo provisto para este fin por el profesor,
alterando las lineas de codigo pertinentes se logró alcanzar una modulacion digital correcta del tipo QPSK, se procederá 
a realizar una breve explicacion del codigo y los resultados obtenidos.

Se importan todas las bibliotecas de python a utilizar

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
```

Las funciones fuente_info y rgb_a_bit no se le realizaron cambios

```python
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
```

Al realizarse la modulación digital QPSK se decidio crear dos vectores que contengan los bits pares e impares respectivamente,
esto con el fin de ingresar cada seccion o vector por un proceso con una portadora diferente, posteriormente se crea la señal
que será transmitida como la union de los dos vectores creados con su respectiva portadora.

```python
def modulador(bits, fc, mpp):
    '''Un método que simula el esquema de 
    modulación digital QPSK.

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
        
   
    # 4. Asignar las formas de onda según los bits (QPSK)
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
```
Para el demodulador se debio realizar modificaciones al codigo debido a la misma situacion que el modulador, existen
dos tipos de portadora que poseen diferente fase al ser una coseno y otra seno, por tanto se necesita capturar las partes que corresponden
a cada una de las portadoras con el fin de realizar una correcta demodulacion, luego se realiza el cambio de la señal en bit a una imagen generada por RGB

```python

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
```

Al realizarse las graficas utilizando las funciones provistas por el profesor se obtuvieron las siguientes imagenes.
![Imagen Recuperada](https://github.com/jorgedaniel-rodriguez/Proyecto4/blob/main/imagenes.png)
![Graficas](https://github.com/jorgedaniel-rodriguez/Proyecto4/blob/main/graficas.png)
 
El error en la imagen se considera bajo debido a que alcanza un valor de 2616 errores, para un BER de 0.0062; como se puede observar
dentro de la señal sinosoidal.

Para obtener datos validos sobre la Estacionaridad y Ergodicidad se utiliza el siguiente codigo y se obtiene n¿una grafica.

```python
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
```
![Graficas](https://github.com/jorgedaniel-rodriguez/Proyecto4/blob/main/estacionaridad.png)

En la grafica se puede observar que los datos obtenidos de la señal y por el hecho de estar conformada por dos portadoras sinosoidales se puede
afirmar que la media se expresa sin dejar a dudas en cero, esto se debe a que la curva en la grafica estan centradas en cero, por tanto la media de valores obtenidos será cero debido
a la estructura con la que esta conformada las funciones sinosoidales y su periodo, otro dato a tomar en cuenta es que dado que la frecuencia en el sistema es constante y su media se encuentra centrada en cero,
el estado estacionario provocara una varianza nula o constante.

Para obtener la densidad espectrar y su grafica se ultiliza el siguiente codigo

```python


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
```
![Densidad Espectral](https://github.com/jorgedaniel-rodriguez/Proyecto4/blob/main/densidad.png)


