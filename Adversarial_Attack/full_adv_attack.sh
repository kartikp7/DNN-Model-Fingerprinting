#!/bin/bash

# usage: ./scripting-auto-a2a-pruning.sh <abs_path_to_venv_with_art_nncf>

art_nncf_venv_path=$1

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

ENSEMBLE_MODEL_FAMILY=("2 VGG16,VGG19 3000" "3 DenseNet121,DenseNet169,DenseNet201 2000" 
"3 ResNet50,ResNet101,ResNet152 2000" "3 MobileNet,MobileNetV2,MobileNetV3Large 2000" 
"3 MobileNet,ResNet50,EfficientNetB4 2000" "3 MobileNet,DenseNet121,EfficientNetB4 2000" 
"3 MobileNet,VGG19,EfficientNetB4 2000")

ENSEMBLE_MODEL_FAMILY_NAME=("VGGs" "DenseNets" "ResNets" "MobileNets" "Mix(w resnet)" "Mix(w densenet)" 
"Mix(w vgg)")

VICTIM_MODEL=(DenseNet121)

ATTACK_NAME=(Deepfool)

DATASET_INDEX=0

MODEL_INDEX_START=0
MODEL_INDEX_END=6

VICTIM_MODEL_INDEX_START=0
VICTIM_MODEL_INDEX_END=0

ATTACK_INDEX_START=0
ATTACK_INDEX_END=0

CTIME="`date +%b-%d-%Y-%H-%M-%p`" 

for (( VICTIM_INDEX=${VICTIM_MODEL_INDEX_START}; VICTIM_INDEX<${VICTIM_MODEL_INDEX_END}+1; VICTIM_INDEX++ ))
do
    PRINT_OUTPUT_FILE="output-${DATASET[$DATASET_INDEX]}-${VICTIM_MODEL[VICTIM_INDEX]}-${CTIME}.log"
    echo $PRINT_OUTPUT_FILE
    #echo "Dataset: ${DATASET[$DATASET_INDEX]}, Fed_training: ${N_TRAINING_SAMPLES}, Fed_testing: ${N_TESTING_SAMPLES}, Fed_adv: ${N_ADV_SAMPLES}, Model: ${MODEL_NAME[$MODEL_INDEX]} ***** Model Start ***** " > $PRINT_OUTPUT_FILE

    #CLASSIFIER_FILE_PREFIX="classifier-${MODEL_NAME[$MODEL_INDEX]}-${DATASET[$DATASET_INDEX]}-on-${N_TRAINING_SAMPLES}"

    for (( ATTACK_INDEX=${ATTACK_INDEX_START}; ATTACK_INDEX<${ATTACK_INDEX_END}+1; ATTACK_INDEX++ ))
    do
        
        for (( MODEL_INDEX=${MODEL_INDEX_START}; MODEL_INDEX<${MODEL_INDEX_END}+1; MODEL_INDEX++ ))
        do
          CLASSIFIER_FILE_PREFIX="classifier-${VICTIM_MODEL[VICTIM_INDEX]}-${DATASET[$DATASET_INDEX]}-on-${N_TRAINING_SAMPLES}"
          echo -e "\t\t\tEnsemble Family: ${ENSEMBLE_MODEL_FAMILY_NAME[$MODEL_INDEX]} on Victim Model: ${VICTIM_MODEL[VICTIM_INDEX]} with Attack: ${ATTACK_NAME[$ATTACK_INDEX]} ***** Model Start ***** " >> $PRINT_OUTPUT_FILE
          #echo -e "\n\nFed_adv: ${N_ADV_SAMPLES}, Attack: ${ATTACK_NAME[$ATTACK_INDEX]} ***** Attack Start ***** " >> $PRINT_OUTPUT_FILE
          source "${art_nncf_venv_path}bin/activate"
          python3 join_adv_examples.py ${ENSEMBLE_MODEL_FAMILY[MODEL_INDEX]} ${DATASET[$DATASET_INDEX]} ${ATTACK_NAME[$ATTACK_INDEX]} ${CLASSIFIER_FILE_PREFIX} ${VICTIM_MODEL[VICTIM_INDEX]} >> $PRINT_OUTPUT_FILE
          deactivate
          echo -e "\n\n" >> $PRINT_OUTPUT_FILE
        done
        echo -e "\n-----------------\n" >> $PRINT_OUTPUT_FILE
    done
done
echo "Done!"