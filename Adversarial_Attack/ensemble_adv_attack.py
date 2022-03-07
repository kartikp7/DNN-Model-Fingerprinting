import matplotlib as mpl
import os
from os import listdir
import time
from numpy.core.fromnumeric import shape
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.framework import convert_to_constants
import sys
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from utility_process_results import process_results
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,BatchNormalization,Activation,Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD

import sys
import numpy as np
from nncf import NNCFConfig
from nncf.tensorflow import create_compressed_model, register_default_init_args
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50

# original_stdout = sys.stdout 
# f = open("output-tensorrt-results.txt", "a")
# sys.stdout = f
print('# '*50+str(time.ctime())+' :: ensemble-attack-results')

time_weight=1000
tmpdir = os.getcwd()
dataset_name=str(sys.argv[1])
model_name=str(sys.argv[2])
attack_name=str(sys.argv[3])
n_test_adv_samples_subset=int(sys.argv[4])
keras_file_name=str(sys.argv[5])
json_path = str(sys.argv[6])
batch_size = int(sys.argv[7])

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
  tf.config.experimental.set_memory_growth(device, True)
# dataset_name+'-x-test-to-tensorrt-'+str(n_test_adv_samples_subset)

## VGG family ensemble examples
x_test_vgg16 = np.load(dataset_name+'-x-test-to-tensorrt-'+str(n_test_adv_samples_subset)+'.npy')
y_test_vgg16 = np.load(dataset_name+'-y-test-to-tensorrt-'+str(n_test_adv_samples_subset)+'.npy')
x_test_adv = np.load(dataset_name+'-'+model_name+'-'+attack_name+'-x-test-adv-to-tensorrt-'+str(n_test_adv_samples_subset)+'.npy')

x_test = np.load(dataset_name+'-x-test-to-tensorrt-'+str(n_test_adv_samples_subset)+'.npy')
y_test = np.load(dataset_name+'-y-test-to-tensorrt-'+str(n_test_adv_samples_subset)+'.npy')
x_test_adv = np.load(dataset_name+'-'+model_name+'-'+attack_name+'-x-test-adv-to-tensorrt-'+str(n_test_adv_samples_subset)+'.npy')

nncf_config = NNCFConfig.from_json(json_path)

(train_images, train_labels), _ = cifar10.load_data()
train_images = train_images / 255.0
print("Train image object: {}, Train label object: {}".format(train_images.shape, train_labels.shape))
train_labels = to_categorical(train_labels)
print("Train image object: {}, Train label object: {}".format(train_images.shape, train_labels.shape))

train_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(train_images, tf.float32),tf.cast(train_labels, tf.int64)))
train_dataset = train_dataset.batch(batch_size)

pretrained_model = tf.keras.models.load_model(keras_file_name+'.h5')
nncf_config = register_default_init_args(nncf_config, train_dataset, batch_size=batch_size)
compression_ctrl, compressed_model = create_compressed_model(pretrained_model, nncf_config)

input_tensor=tf.constant(x_test.astype('float32'))

compressed_model.predict(input_tensor)
start_time=time.time()
predictions = compressed_model.predict(input_tensor)
end_time = time.time()
elapsed_time = end_time - start_time
print("NNCF stats on benign test examples with inference in {:.2f} ms.".format(elapsed_time*time_weight))
process_results(predictions, y_test)

input_tensor=tf.constant(x_test_adv.astype('float32'))
start_time=time.time()
predictions = compressed_model.predict(input_tensor)
end_time = time.time()
elapsed_time = end_time - start_time
print("NNCF stats on adversarial test examples with inference in {:.2f} ms.".format(elapsed_time*time_weight))
process_results(predictions, y_test)
optimization_str = json_path.split('/')[-1].split('.json')[0]