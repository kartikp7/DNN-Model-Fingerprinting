import tensorflow as tf
import numpy as np
import sys
import time
from utility_process_results import process_results

time_weight=1000

ensemble_size=int(sys.argv[1])
list_of_models=sys.argv[2].split(",")
n_test_samples=int(sys.argv[3])
# model_family_name=str(sys.argv[3])
dataset_name=str(sys.argv[4])
attack_name=str(sys.argv[5])
keras_file_name=str(sys.argv[6]) # victim model
victim_model_name=str(sys.argv[7]) # victim model name

# print("Inputs to py file:")
# print("ensemble_size: ", ensemble_size)
# print("list_of_models: ",list_of_models)

# for model in list_of_models:
#   print("   ", model)

# print("n_test_samples: ",n_test_samples)
# print("dataset_name: ",dataset_name)
# print("attack_name: ", attack_name)
# print("keras_file_name: ", keras_file_name)
# print("victim_model_name: ",victim_model_name)
# print("-"*20)

# first model in list
x_test_adv_global = np.load(dataset_name+'-'+list_of_models[0]+'-'+attack_name+'-x-test-adv-'+str(n_test_samples)+'.npy')
x_test_global = np.load(dataset_name+'-'+list_of_models[0]+'-'+attack_name+'-x-test-'+str(n_test_samples)+'.npy')
y_test_global = np.load(dataset_name+'-'+list_of_models[0]+'-'+attack_name+'-y-test-'+str(n_test_samples)+'.npy')

for i in range(ensemble_size):
  if i == 0:
    continue
  x_test_adv_local = np.load(dataset_name+'-'+list_of_models[i]+'-'+attack_name+'-x-test-adv-'+str(n_test_samples)+'.npy')
  x_test_local = np.load(dataset_name+'-'+list_of_models[i]+'-'+attack_name+'-x-test-'+str(n_test_samples)+'.npy')
  y_test_local = np.load(dataset_name+'-'+list_of_models[i]+'-'+attack_name+'-y-test-'+str(n_test_samples)+'.npy')

  x_test_global = np.concatenate((x_test_global, x_test_local))
  y_test_global = np.concatenate((y_test_global, y_test_local))
  x_test_adv_global = np.concatenate((x_test_adv_global, x_test_adv_local))

# Victim model
classifier = tf.keras.models.load_model(keras_file_name+'.h5')

# Pred with Benign Examples
start_time = time.time()
predictions = classifier.predict(x_test_global)
end_time = time.time()
elapsed_time = end_time - start_time
print("Victim model "+str(victim_model_name)+" with benign test examples with inference in {:.2f} ms.".format(elapsed_time*time_weight))
process_results(predictions, y_test_global)

# Pred on Adversarial Examples
# classifier.predict(x_test_adv_global)
start_time = time.time()
predictions = classifier.predict(x_test_adv_global)
end_time = time.time()
elapsed_time = end_time - start_time
print("Victim model "+str(victim_model_name)+" with adversarial test examples with inference in {:.2f} ms.".format(elapsed_time*time_weight))
process_results(predictions, y_test_global)