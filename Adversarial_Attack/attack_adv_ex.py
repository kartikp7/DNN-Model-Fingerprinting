from art.attacks.attack import Attack
from art.attacks.evasion.auto_attack import AutoAttack
from art.attacks.evasion.carlini import CarliniL0Method, CarliniLInfMethod, CarliniL2Method
from art.attacks.evasion import DeepFool, FastGradientMethod, AutoProjectedGradientDescent, ShadowAttack, Wasserstein, BrendelBethgeAttack, ShapeShifter, UniversalPerturbation, NewtonFool
from art.attacks.evasion.iterative_method import BasicIterativeMethod
from art.attacks.evasion.elastic_net import ElasticNet
from art.attacks.evasion.adversarial_patch.adversarial_patch import AdversarialPatch
from art.attacks.evasion.targeted_universal_perturbation import TargetedUniversalPerturbation
from utility_process_results import process_results
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np

from art.estimators.classification import KerasClassifier
from art.utils import load_mnist, to_categorical
from art.utils import load_cifar10
from tests.utils import get_image_classifier_tf, get_image_classifier_kr
from tensorflow.keras import backend as K
import time

import warnings
#suppress warnings
warnings.filterwarnings('ignore')

###############
import matplotlib as mpl
import os
from os import listdir
import time
# import tensorflow as tf
# import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.framework import convert_to_constants
import sys


print('# '*50+str(time.ctime())+' :: smh-attack-and-adv-examples')

ATTACK_NAME={"CW":"CarliniWagner", "DF":"Deepfool", "FGSM":"FastGradientMethod", "APGD":"AutoProjectedGradientDescent", "SA":"ShadowAttack", "WS":"Wasserstein",
            "EN":"ElasticNet","ADP":"AdversarialPatch", "BIM":"BasicIterativeMethod", "UP":"UniversalPerturbation","NF":"NewtonFool","TUP":"TargetedUniversalPerturbation"}
tmpdir = os.getcwd()
time_weight=1000
# # Step 2: Load Model
keras_file_name=str(sys.argv[1])
dataset_name=str(sys.argv[2])
model_name=str(sys.argv[3])
attack_name=str(sys.argv[4])
n_test_samples=int(sys.argv[5])
batch_size = int(sys.argv[6])
K.clear_session()
model = tf.keras.models.load_model(keras_file_name+'.h5')
x_test = np.load(dataset_name+'-'+model_name+'-'+attack_name+'-x-test-'+str(n_test_samples)+'.npy')
y_test = np.load(dataset_name+'-'+model_name+'-'+attack_name+'-y-test-'+str(n_test_samples)+'.npy')


classifier = KerasClassifier(model=model,clip_values=(0, 1))#, clip_values=(min_pixel_value, max_pixel_value), use_logits=False
classifier.predict(x_test)
start_time=time.time()
predictions = classifier.predict(x_test)
end_time = time.time()
elapsed_time = end_time - start_time
print("Full-bone stats on benign test examples with inference in {:.2f} ms.".format(elapsed_time*time_weight))
process_results(predictions, y_test)


# Step 6: Generate adversarial test examples
flag_TUP_Attack=False
if attack_name==ATTACK_NAME.get("CW"):
    attack = CarliniL2Method(classifier=classifier, max_iter=10, batch_size=batch_size, verbose=True)
elif attack_name==ATTACK_NAME.get("DF"):
    attack = DeepFool(classifier=classifier, max_iter=5, batch_size=batch_size, verbose=True)
elif attack_name==ATTACK_NAME.get("FGSM"):
    attack = FastGradientMethod(estimator=classifier, eps=0.2)
elif attack_name==ATTACK_NAME.get("EN"):
    attack = ElasticNet(classifier=classifier,targeted=False, max_iter=2, verbose=True)
elif attack_name==ATTACK_NAME.get("ADP"):
    attack = AdversarialPatch(classifier=classifier,rotation_max=0.5, scale_min=0.4, scale_max=0.41, learning_rate=5.0, batch_size=batch_size, max_iter=5, verbose=True)
elif attack_name==ATTACK_NAME.get("NF"):
    attack = NewtonFool(classifier,max_iter=5, batch_size=batch_size, verbose=True)
elif attack_name==ATTACK_NAME.get("BIM"):
    attack = BasicIterativeMethod(classifier,eps=1.0, eps_step=0.1, batch_size=batch_size, verbose=True)
elif attack_name==ATTACK_NAME.get("UP"):
    attack = UniversalPerturbation(classifier,max_iter=1,attacker="ead",attacker_params={"max_iter": 2, "targeted": False, "verbose": True},verbose=True)
elif attack_name==ATTACK_NAME.get("TUP"):
    attack = TargetedUniversalPerturbation(classifier,max_iter=1, attacker="fgsm", attacker_params={"eps": 0.3, "targeted": True, "verbose": True})
    target = 0
    y_target = np.zeros([len(x_test), 10])
    for i in range(len(x_test)):
        y_target[i, target] = 1.0
    flag_TUP_Attack=True
elif attack_name==ATTACK_NAME.get("WS"):
    attack = Wasserstein(classifier,regularization=100,conjugate_sinkhorn_max_iter=5, projected_sinkhorn_max_iter=5,norm="wasserstein",ball="wasserstein",targeted=False,p=2,eps_iter=2,eps_factor=1.05,eps_step=0.1,kernel_size=5,batch_size=batch_size,verbose=True)

start_time=time.time()
if flag_TUP_Attack==True:
    x_test_adv = attack.generate(x_test,y=y_target)
else:
    x_test_adv = attack.generate(x_test)
end_time = time.time()
elapsed_time = end_time - start_time
print("Adversarial examples generation time: {:.2f} ms.".format(elapsed_time*time_weight))
# Step 7: Evaluate the ART classifier on adversarial test examples

np.save(dataset_name+'-'+model_name+'-'+attack_name+'-x-test-adv-'+str(n_test_samples),x_test_adv)
classifier.predict(x_test_adv)
start_time = time.time()
predictions = classifier.predict(x_test_adv)
end_time = time.time()
elapsed_time = end_time - start_time
print("Full-bone stats on adversarial test examples with inference in {:.2f} ms.".format(elapsed_time*time_weight))
process_results(predictions, y_test)