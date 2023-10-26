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




import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, LeakyReLU, Add, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


class SelfAttention(tf.keras.layers.Layer):

    """
    Explicación de la clase que implementa la capa de atención:
        __init__(self, filters): Este es el constructor de la clase. Recibe un parámetro llamado filters, que determina el número de filtros (características) que se utilizarán para calcular la atención. Esta cantidad de filtros influye en cuántas características se considerarán relevantes.

        build(self, input_shape): Aquí se construye la capa. Se crea una serie de pesos (theta, phi y g) que se utilizarán para calcular el mapa de atención. Estos pesos son matrices que se aplicarán a la entrada para extraer características relevantes.

        call(self, inputs): Esta función es la esencia de la capa de atención. Toma la entrada inputs, que es la imagen a la que queremos aplicar la atención. Luego, se realiza una serie de cálculos utilizando los pesos (theta, phi y g) para obtener el mapa de atención.

        theta, phi y g: Son los pesos que se aprenden durante el entrenamiento de la red neuronal. Estos pesos se utilizan para transformar la entrada y calcular el mapa de atención.

        theta_phi_dot = tf.matmul(theta, phi, transpose_b=True): Aquí se realiza una operación de multiplicación de matrices entre los mapas theta y phi, transponiendo uno de ellos. Esto da como resultado un mapa que representa las conexiones entre diferentes partes de la imagen.

        attention_map = tf.nn.softmax(theta_phi_dot): Se aplica una función de softmax al mapa de conexiones para obtener el mapa de atención final. Esta función de softmax normaliza las conexiones y asegura que las regiones más relevantes tengan valores más altos en el mapa de atención.

        theta_phi_g_dot = tf.matmul(attention_map, g): Ahora, el mapa de atención se utiliza para calcular un nuevo mapa que resalta las características más importantes de la imagen.

        out = self.out_conv(theta_phi_g_dot): Finalmente, el mapa de características resaltadas se pasa a través de una convolución con un filtro de 1x1 para ajustar la dimensión de las características y obtener el resultado final.

        out = Add()([out, inputs]): Finalmente, las características resaltadas se suman a las características originales de la entrada, lo que permite que la información relevante y original coexistan en la salida  
    
    """
    
    def __init__(self, filters):
        super(SelfAttention, self).__init__()
        self.filters = filters

    def build(self, input_shape):
        self.theta = self.add_weight(name='theta', shape=(1, 1, input_shape[3], self.filters), initializer='glorot_uniform', trainable=True)
        self.phi = self.add_weight(name='phi', shape=(1, 1, input_shape[3], self.filters), initializer='glorot_uniform', trainable=True)
        self.g = self.add_weight(name='g', shape=(1, 1, input_shape[3], self.filters), initializer='glorot_uniform', trainable=True)
        self.out_conv = Conv2D(input_shape[3], (1, 1), strides=(1, 1), padding='same')

    def call(self, inputs):
        theta = self.conv_block(inputs, self.theta)
        phi = self.conv_block(inputs, self.phi)
        g = self.conv_block(inputs, self.g)

        theta_phi_dot = tf.matmul(theta, phi, transpose_b=True)
        attention_map = tf.nn.softmax(theta_phi_dot)

        theta_phi_g_dot = tf.matmul(attention_map, g)
        out = self.out_conv(theta_phi_g_dot)

        out = Add()([out, inputs])
        return out

    def conv_block(self, x, weights):
        conv = tf.nn.conv2d(x, weights, strides=(1, 1), padding='SAME')
        return conv


def TL_unet_model(input_shape, num_classes):
    # input: input_shape (height, width, channels)
    # return model
    input_shape = input_shape
    base_VGG = VGG16(include_top=False,
                     weights="imagenet",
                     input_shape=input_shape)

    # freezing all layers in VGG16
    for layer in base_VGG.layers:
        layer.trainable = True

    # the bridge (exclude the last maxpooling layer in VGG16)
    bridge = base_VGG.get_layer("block5_conv3").output

    # Add self-attention to the bridge
    bridge = SelfAttention(512)(bridge)

    # Decoder now
    up1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bridge)
    concat_1 = concatenate([up1, base_VGG.get_layer("block4_conv3").output], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(concat_1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    concat_2 = concatenate([up2, base_VGG.get_layer("block3_conv3").output], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(concat_2)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    concat_3 = concatenate([up3, base_VGG.get_layer("block2_conv2").output], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat_3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    concat_4 = concatenate([up4, base_VGG.get_layer("block1_conv2").output], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat_4)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

    model_ = Model(inputs=[base_VGG.input], outputs=[conv10])

    model_.summary()

    return model_




from tensorflow import keras
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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred, smooth=1.):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    sum_ = K.sum(y_true + y_pred, axis=[1, 2, 3])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def compile_model(model):
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=dice_coef_loss,
                  metrics=[Accuracy(), MeanIoU(num_classes=num_classes), dice_coef])

def train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs):
    checkpoint = ModelCheckpoint("last_VGGUnet.h5", monitor='val_dice_coef', save_best_only=True, mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.1, patience=15, min_lr=1e-6, mode='max', verbose=1)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs,callbacks=[checkpoint, reduce_lr])




input_shape = (192, 192, 3)
num_classes = 13


model = TL_unet_model(input_shape, num_classes)

# Compile the model
compile_model(model)

#wandb.login(key='125ad0638d8ab092fd2e02f938ff7040ac7e1e90')  # Reemplaza 'tu_clave_de_api' con tu propia clave de API

# wandb.init(project='VGG16-Unet', entity='boresmol')

# Train the model
history = train_model(model, imagenes_train, mascaras_train, imagenes_val, mascaras_val, batch_size=1, epochs=12)

model.save('AttentionUnet.h5')

#wandb.log({'Training Loss': history.history['loss'], 'Validation Loss': history.history['val_loss']})
#wandb.log({'Training Mean IoU': history.history['mean_io_u'], 'Validation Mean IoU': history.history['val_mean_io_u']})
#wandb.log({'Training Accuracy': history.history['accuracy'], 'Validation Accuracy': history.history['val_accuracy']})
#wandb.log({'Training Dice Coef': history.history['dice_coef'], 'Validation Dice Coef': history.history['val_dice_coef']})