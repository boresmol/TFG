import os
import subprocess
import sys


subprocess.check_call([sys.executable, "-m", "pip", "install", "np"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow_addons"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])


subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "keras"])

subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "tensorflow"])

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import confusion_matrix
from wandb.keras import WandbCallback
import tqdm
from custom_loss import dice_coef,dice_coef_loss


import numpy as np
import psutil
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, concatenate
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import Accuracy, MeanIoU
import tensorflow.keras.backend as K

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, concatenate
from tensorflow.keras.applications import VGG16

import wandb
from wandb.keras import WandbCallback



# Función para cargar el modelo Keras (.h5)
def load_model_with_custom_loss(model_path):
    custom_objects = {'dice_coef_loss': dice_coef_loss}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)



# Función para hacer predicciones en el conjunto de imágenes test
def predict_images(model, test_images):
    return model.predict(test_images)

# Función para dibujar la matriz de confusión y enviarla a wandb
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    return cm

# Función para dibujar la segmentación de la imagen por colores y enviarla a wandb
def plot_segmentation(image, mask, class_colors):
    segmentation = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for i in range(mask.shape[0]):
        mask_class = np.argmax(mask[i], axis=-1)
        for j in range(len(class_colors)):
            segmentation[mask_class == j] = class_colors[j]

    plt.imshow(segmentation)
    plt.title("Segmentation")
    plt.axis("off")
    plt.show()

# Función para calcular el mean IoU (Intersection over Union)
def mean_iou(y_true, y_pred, smooth=1.):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    sum_ = K.sum(y_true + y_pred, axis=[1, 2, 3])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


model = load_model_with_custom_loss('vggUnetCustomLoss.h5')


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

print('División en diferentes sets')
# Dividir los conjuntos de datos en train, val y test
imagenes_train_val, imagenes_test, mascaras_train_val, mascaras_test = train_test_split(
    imagenes_total, mascaras_total, test_size=0.2, random_state=42)
imagenes_train, imagenes_val, mascaras_train, mascaras_val = train_test_split(
    imagenes_train_val, mascaras_train_val, test_size=0.25, random_state=42)

# Verificar las formas de los conjuntos de datos generados
print("Forma de imágenes_train:", imagenes_train.shape)
print("Forma de imágenes_val:", imagenes_val.shape)
print("Forma de imágenes_test:", imagenes_test.shape)
print("Forma de mascaras_train:", mascaras_train.shape)
print("Forma de mascaras_val:", mascaras_val.shape)
print("Forma de mascaras_test:", mascaras_test.shape)



# Hacer predicciones en el conjunto de imágenes test
predictions = predict_images(model, imagenes_test)


wandb.login(key='125ad0638d8ab092fd2e02f938ff7040ac7e1e90')  # Reemplaza 'tu_clave_de_api' con tu propia clave de API

wandb.init(project='evaluacion', entity='boresmol')


# Dibujar y enviar la matriz de confusión a wandb
class_names = ['wooded', 'agricultural', 'residential', 'industrial','comercial','recreation','airport','quarry','military','desert_sand','mountain_rock','other_natural','unlabeled']  # Sustituye con los nombres de las clases reales
confusion_matrix = plot_confusion_matrix(mascaras_test, predictions, class_names)
wandb.log({"Confusion Matrix": wandb.plot.confusion_matrix(
    y_true=mascaras_test.argmax(axis=-1),
    preds=predictions.argmax(axis=-1),
    class_names=class_names
)})

# Dibujar y enviar la segmentación de la imagen a wandb
class_colors = [
    [255, 0, 0],   # Rojo
    [0, 255, 0],   # Verde
    [0, 0, 255],   # Azul
    [255, 255, 0],   # Amarillo
    [255, 0, 255],   # Magenta
    [0, 255, 255],   # Cian
    [128, 0, 0],   # Marrón oscuro
    [0, 128, 0],   # Verde oscuro
    [0, 0, 128],   # Azul oscuro
    [128, 128, 0],   # Amarillo oscuro
    [128, 0, 128],   # Magenta oscuro
    [0, 128, 128],   # Cian oscuro
    [128, 128, 128]  # Gris
]
 # Sustituye con los colores de cada clase
segmentation = plot_segmentation(mascaras_test[0], predictions[0], class_colors)
wandb.log({"Segmentation": wandb.Image(np.uint8(segmentation))})

# Calcular y mostrar el accuracy, mean IoU y Dice Coefficient
accuracy = np.mean(np.argmax(predictions, axis=-1) == np.argmax(mascaras_test, axis=-1))
mean_iou_value = mean_iou(mascaras_test, np.round(predictions))
dice_coef_value = dice_coef(mascaras_test, np.round(predictions))

print("Accuracy:", accuracy)
print("Mean IoU:", mean_iou_value)
print("Dice Coefficient:", dice_coef_value)



