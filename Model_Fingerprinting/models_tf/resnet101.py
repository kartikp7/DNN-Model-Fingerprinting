import time
import os
import random
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications.resnet import decode_predictions

def main(argv):

    if len(sys.argv) < 2:
        print("Error: Missing argument - test dataset")
        exit(0)

    test_dir = sys.argv[1]
    
    BATCH_SIZE = 1
    IMG_SIZE = (224, 224)

    test_img = random.choice(os.listdir(test_dir))
    test_img_pth = os.path.join(test_dir, test_img)
    img = load_img(test_img_pth, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    
    model = ResNet101(weights='imagenet')
    output = model.predict(img)
    label = decode_predictions(output)
    label = label[0][0]
    # print(label[1])


if __name__ == "__main__":
    main(sys.argv[1:])
