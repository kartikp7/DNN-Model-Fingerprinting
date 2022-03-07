import tensorflow as tf
from tensorflow.python.keras.activations import get
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.backend import flatten

tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization,Activation,Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152, VGG19, VGG16, MobileNet, MobileNetV2, MobileNetV3Large, DenseNet121, DenseNet169, DenseNet201, InceptionV3, EfficientNetB4
import numpy as np

from art.estimators.classification import KerasClassifier
from art.utils import load_mnist, to_categorical
from art.utils import load_cifar10

###############
import matplotlib as mpl
import os
from os import listdir
import time
from matplotlib import pyplot as plt
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.framework import convert_to_constants
import sys

import argparse

print('# '*50+str(time.ctime()))
tmpdir = os.getcwd()
time_weight=1000
MODEL_NAME={"VGG19":"VGG19", "VGG16":"VGG16", "RESNET50":"ResNet50", "RESNET101":"ResNet101", "RESNET152":"ResNet152", 
"MOBILENET": "MobileNet", "MOBILENETV2": "MobileNetV2", "MOBILENETV3LARGE": "MobileNetV3Large", "DENSENET121":"DenseNet121", 
"DENSENET169":"DenseNet169", "DENSENET201":"DenseNet201", "INCEPTION":"InceptionV3", "EFFICIENTNETB4":"EfficientNetB4"}
(x_train, y_train), _, min_pixel_value, max_pixel_value = load_cifar10()

n_train_samples = int(sys.argv[1])
batch_size = int(sys.argv[2])
nb_epochs= int(sys.argv[3])
keras_file_name=str(sys.argv[4])
model_name=str(sys.argv[5])

x_train = x_train[:n_train_samples]
y_train = y_train[:n_train_samples]
print((x_train.shape,y_train.shape))

# # model= Sequential()
# # Step 2: Load Model and do transfer learning 
if model_name==MODEL_NAME.get("VGG19"):
    backbone = VGG19(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
elif model_name==MODEL_NAME.get("VGG16"):
    backbone = VGG16(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
elif model_name==MODEL_NAME.get("RESNET50"):
    backbone = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
elif model_name==MODEL_NAME.get("RESNET101"):
    backbone = ResNet101(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
elif model_name==MODEL_NAME.get("RESNET152"):
    backbone = ResNet152(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
elif model_name == MODEL_NAME.get("MOBILENET"):
    backbone = MobileNet(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
elif model_name == MODEL_NAME.get("MOBILENETV2"):
    backbone = MobileNetV2(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
elif model_name == MODEL_NAME.get("MOBILENETV3LARGE"):
    backbone = MobileNetV3Large(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
elif model_name == MODEL_NAME.get("DENSENET121"):
    backbone = DenseNet121(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
elif model_name == MODEL_NAME.get("DENSENET169"):
    backbone = DenseNet169(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
elif model_name == MODEL_NAME.get("DENSENET201"):
    backbone = DenseNet201(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
elif model_name == MODEL_NAME.get("INCEPTION"):
    backbone = InceptionV3(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')
elif model_name == MODEL_NAME.get("EFFICIENTNETB4"):
    backbone = EfficientNetB4(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='avg')

output = tf.keras.layers.Dense(y_train.shape[1], activation='softmax', name='predictions')(backbone.output)
model = tf.keras.Model(backbone.input, output)
model.compile(optimizer= 'adam',loss=categorical_crossentropy,metrics=["accuracy"])
classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)

print(model_name + " Training...")
start_time=time.time()
classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs)#, verbose=False
end_time = time.time()
elapsed_time = end_time - start_time
print("{} - batch size : {} - epochs : {} - training in {:.2f} ms.".format(keras_file_name,batch_size,nb_epochs,elapsed_time*time_weight))
classifier.save(keras_file_name+'.h5', tmpdir)