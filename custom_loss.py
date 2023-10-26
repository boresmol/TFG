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