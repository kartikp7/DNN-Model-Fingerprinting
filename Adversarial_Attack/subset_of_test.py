import numpy as np
import sys
from art.utils import load_cifar10
import argparse
import time

print('# '*50+str(time.ctime())+' :: smh-subset-of-test')

per_class_samples = int(sys.argv[1])
n_classes = int(sys.argv[2])
dataset_name=str(sys.argv[3])
model_name=str(sys.argv[4])
attack_name=str(sys.argv[5])

_, (x_test, y_test), _, _ = load_cifar10()
print(x_test.shape)
print(y_test.shape)
x_test_subset = []
y_test_subset = []
count = np.zeros(n_classes)
t=0

# Randomly Shuffle dataset
shuffler = np.random.permutation(len(y_test))
y_test_shuffled = y_test[shuffler]
x_test_shuffled = x_test[shuffler]

for y in y_test_shuffled:
    pos = np.where(y==1)
    count[pos] =  count[pos]+1
    if count[pos]<=per_class_samples:
        x_test_subset.append(x_test_shuffled[t])
        y_test_subset.append(y)
        # print(str(pos)+'::'+str(count[pos]))
    t = t+1
x_test_subset = np.array(x_test_subset)
y_test_subset = np.array(y_test_subset)
print(x_test_subset.shape)
print(y_test_subset.shape)
n_test_samples=n_classes*per_class_samples
np.save(dataset_name+'-'+model_name+'-'+attack_name+'-x-test-'+str(n_test_samples), x_test_subset)
np.save(dataset_name+'-'+model_name+'-'+attack_name+'-y-test-'+str(n_test_samples), y_test_subset)