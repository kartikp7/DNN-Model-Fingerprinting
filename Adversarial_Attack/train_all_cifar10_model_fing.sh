#!/bin/bash

# usage: ./scripting-auto-a2a-pruning.sh <abs_path_to_venv_with_art_nncf> <train - 0/1>

art_nncf_venv_path=$1
train_flag=$2

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$art_library_path

N_PER_CLASS_TRAINING_SAMPLES=5000
N_BATCH_SIZE=128
N_EPOCHS=20

N_PER_CLASS_TESTING_SAMPLES=1000

N_PER_CLASS_ADV_SAMPLES=$N_PER_CLASS_TESTING_SAMPLES

N_CLASSES=10
N_TRAINING_SAMPLES=$(($N_PER_CLASS_TRAINING_SAMPLES*$N_CLASSES))
N_TESTING_SAMPLES=$(($N_PER_CLASS_TESTING_SAMPLES*$N_CLASSES))
N_ADV_SAMPLES=$(($N_PER_CLASS_ADV_SAMPLES*$N_CLASSES))

### DATASET ###
DATASET=(cifar10) # imagenet)

### MODELS ###
#MODEL_NAME=(MobileNet DenseNet121 VGG19 ResNet50) # Already trained
#MODEL_NAME=(DenseNet169 DenseNet201 VGG16 ResNet101 ResNet152 MobileNetV2 MobileNetV3Large EfficientNetB4)
MODEL_NAME=(DenseNet169 VGG16 MobileNetV2 EfficientNetB4 DenseNet201 ResNet101 ResNet152 MobileNetV3Large)


DATASET_INDEX=0

MODEL_INDEX_START=0
MODEL_INDEX_END=7

CTIME="`date +%b-%d-%Y-%H-%M-%p`" 

for (( MODEL_INDEX=${MODEL_INDEX_START}; MODEL_INDEX<${MODEL_INDEX_END}+1; MODEL_INDEX++ ))
do
    PRINT_OUTPUT_FILE="output-${DATASET[$DATASET_INDEX]}-${MODEL_NAME[$MODEL_INDEX]}-${CTIME}.log"
    echo $PRINT_OUTPUT_FILE
    echo "Dataset: ${DATASET[$DATASET_INDEX]}, Fed_training: ${N_TRAINING_SAMPLES}, Fed_testing: ${N_TESTING_SAMPLES}, Fed_adv: ${N_ADV_SAMPLES}, Model: ${MODEL_NAME[$MODEL_INDEX]} ***** Model Start ***** " > $PRINT_OUTPUT_FILE

    CLASSIFIER_FILE_PREFIX="classifier-${MODEL_NAME[$MODEL_INDEX]}-${DATASET[$DATASET_INDEX]}-on-${N_TRAINING_SAMPLES}"
     
    if [ $train_flag -eq 1 ]; then
        source "${art_nncf_venv_path}bin/activate"
        python3 train_clf_model_fing.py $N_TRAINING_SAMPLES $N_BATCH_SIZE $N_EPOCHS $CLASSIFIER_FILE_PREFIX ${MODEL_NAME[$MODEL_INDEX]} >> $PRINT_OUTPUT_FILE
        deactivate
    fi 
    #echo -e "\n ### ${MODEL_NAME[$MODEL_INDEX]} DONE ###\n"

done
echo "Done!"