import numpy as np
import sys

from sklearn.utils import shuffle
from art.utils import load_cifar10
import argparse
import time
import random

dataset_name=str(sys.argv[1])
n_test_samples=int(sys.argv[2])
n_way = int(sys.argv[3]) # 0 is 2-way, 1 is 3-way

x_test = np.load(dataset_name+'-x-test-'+str(n_test_samples)+'.npy')
y_test = np.load(dataset_name+'-y-test-'+str(n_test_samples)+'.npy')

two_way = int(n_test_samples / 2)
three_way = int(n_test_samples / 3)

new_y_test = []
new_x_test = []
check = []

ctr=0
for i, y in enumerate(y_test):
  #print(y, end=" ")
  #pos = np.where(y==1)
  #print(pos, i)
  #new_y_test.append((y, i))
  new_y_test.append(i)

for i, x in enumerate(x_test):
  #print(y, end=" ")
  #pos = np.where(x==1)
  #print(pos, i)
  check.append(i)


array1 = np.array([i for i in range(5)])
array2 = np.array([i for i in range(5,10)])
print()
print("Array1: ", end=" ")
print(array1)
print("Array2: ", end=" ")
print(array2)
print()
print('after shuffle:')

print()

shuffler = np.random.permutation(len(array1))

#y_test_shuffled = y_test[shuffler]
#x_test_shuffled = x_test[shuffler]
array1_shuffled = array1[shuffler]
array2_shuffled = array2[shuffler]

print("Array1 shuffled: ", end=" ")
print(array1_shuffled)
print("Array2 shuffled: ", end=" ")
print(array2_shuffled)
print()
#print(two_way, three_way)


