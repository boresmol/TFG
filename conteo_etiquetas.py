import os

# Obtener el entorno Conda actual
conda_env = os.environ.get('CONDA_DEFAULT_ENV')

# Imprimir el entorno Conda
print("Entorno Conda actual:", conda_env)


import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "np"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow_addons"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])

subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "keras"])

subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "tensorflow"])



print('Comenzando la carga de librerías...')
import numpy as np
import psutil
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm



if tf.test.is_gpu_available():
    print("Se ha detectado una GPU en el sistema.")
else:
    print("No se ha detectado una GPU en el sistema.")

print('Completada carga de librerías')


print('Cargando máscaras...')

archivo_npy = 'mask_array.npy'
mascaras = np.load(archivo_npy)

print('Máscaras cargadas correctamente')
print('El shape de la máscara es:')
print(mascaras.shape)

print('Cargando imágenes ........')


archivo_npy = 'data_imagenes.npy'


imagenes = np.load(archivo_npy, mmap_mode='r')

print('CARGA DE IMÁGENES COMPLETADA !!!!!!')
print('EL SHAPE DEL ARRAY CARGADO ES: ')
# Imprimir los datos
print(imagenes.shape)


print('cambiando valores 255 para el one hot...')

mascaras[mascaras == 255] = 12

print('Valores cambiado, empezando one hot...')

memoria = psutil.virtual_memory()


memoria_utilizada = memoria.used / 1024 / 1024
print(f"Memoria utilizada: {memoria_utilizada:.2f} MB")


num_classes = 13
one_hot_masks = tf.keras.utils.to_categorical(mascaras, num_classes)
print('one hot acabado...')

print('El shape actual de las máscaras es:')
print(one_hot_masks.shape)



# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,  # rango de rotación aleatoria en grados
    width_shift_range=0.1,  # rango de desplazamiento horizontal aleatorio
    height_shift_range=0.1,  # rango de desplazamiento vertical aleatorio
    zoom_range=0.2,  # rango de zoom aleatorio
    horizontal_flip=True  # volteo horizontal aleatorio
)

imagenes_aug = []
mascaras_aug = []

print('Comienza el data augmentation.......')

for i in tqdm(range(imagenes.shape[0]), desc='Data Augmentation'):
    imagen = imagenes[i]
    mascara = one_hot_masks[i]
    imagen_aug = datagen.random_transform(imagen)
    mascara_aug = datagen.random_transform(mascara)
    imagenes_aug.append(imagen_aug)
    mascaras_aug.append(mascara_aug)



imagenes_aug = np.array(imagenes_aug)
mascaras_aug = np.array(mascaras_aug)

print('Concatenación con las imagenes originales')

# Concatenar las imágenes y máscaras originales con las aumentadas
imagenes_total = np.concatenate((imagenes, imagenes_aug), axis=0)
mascaras_total = np.concatenate((one_hot_masks, mascaras_aug), axis=0)

print(f'Imagenes de data aug: {imagenes_aug.shape}')
print(f'Mascaras de data aug: {mascaras_aug.shape}')
print(f'Imagenes concatenadas {imagenes_total.shape}')
print(f'Mascaras concatewndas: {mascaras_total.shape}')




import pandas as pd

# Supongamos que ya has cargado tus máscaras y realizado las operaciones necesarias

# Obtener la suma de etiquetas por clase
conteo_etiquetas = mascaras_total.sum(axis=(0, 1, 2))

# Crear un DataFrame de pandas para el conteo de etiquetas
df = pd.DataFrame({'Clase': range(num_classes), 'Cantidad': conteo_etiquetas})

# Guardar el conteo de etiquetas en un archivo Excel
archivo_excel = 'conteo_etiquetas.xlsx'
df.to_excel(archivo_excel, index=False)

print(f"Conteo de etiquetas guardado en '{archivo_excel}'")
