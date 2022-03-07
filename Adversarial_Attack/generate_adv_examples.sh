#!/bin/bash

# usage: ./scripting-auto-a2a-pruning.sh <abs_path_to_venv_with_art_nncf> <train - 0/1> <attack_flag - 0/1>

art_nncf_venv_path=$1
train_flag=$2
attack_flag=$3

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$art_library_path

N_PER_CLASS_TRAINING_SAMPLES=5000
N_BATCH_SIZE=128
N_EPOCHS=20

# 300 for VGGs, 200 for rest
N_PER_CLASS_TESTING_SAMPLES=300

N_PER_CLASS_ADV_SAMPLES=$N_PER_CLASS_TESTING_SAMPLES

N_CLASSES=10
N_TRAINING_SAMPLES=$(($N_PER_CLASS_TRAINING_SAMPLES*$N_CLASSES))
N_TESTING_SAMPLES=$(($N_PER_CLASS_TESTING_SAMPLES*$N_CLASSES))
N_ADV_SAMPLES=$(($N_PER_CLASS_ADV_SAMPLES*$N_CLASSES))
TRT_INPUT_1D=32

DATASET=(cifar10)

# MODEL_NAME=(MobileNetV2 DenseNet169 EfficientNetB4 ResNet152)

# MODEL_NAME=(DenseNet121 DenseNet201 ResNet50 ResNet101 MobileNet MobileNetV3Large)

MODEL_NAME=(VGG16 VGG19)

ATTACK_NAME=(Deepfool)

DATASET_INDEX=0

MODEL_INDEX_START=0
MODEL_INDEX_END=1

ATTACK_INDEX_START=0
ATTACK_INDEX_END=1

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

    for (( ATTACK_INDEX=${ATTACK_INDEX_START}; ATTACK_INDEX<${ATTACK_INDEX_END}+1; ATTACK_INDEX++ ))
    do
        echo -e "\n\nFed_adv: ${N_ADV_SAMPLES}, Attack: ${ATTACK_NAME[$ATTACK_INDEX]} ***** Attack Start ***** " >> $PRINT_OUTPUT_FILE
        if [ $attack_flag -eq 1 ]; then
            source "${art_nncf_venv_path}bin/activate"
            python3 subset_of_test.py $N_PER_CLASS_TESTING_SAMPLES $N_CLASSES ${DATASET[$DATASET_INDEX]} ${MODEL_NAME[$MODEL_INDEX]} ${ATTACK_NAME[$ATTACK_INDEX]} >> $PRINT_OUTPUT_FILE
            python3 attack_adv_ex.py $CLASSIFIER_FILE_PREFIX ${DATASET[$DATASET_INDEX]} ${MODEL_NAME[$MODEL_INDEX]} ${ATTACK_NAME[$ATTACK_INDEX]} $N_TESTING_SAMPLES $N_BATCH_SIZE >> $PRINT_OUTPUT_FILE
            deactivate
        fi 
    done
done
echo "Done!"