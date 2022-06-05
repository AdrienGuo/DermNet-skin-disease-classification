from dataset import dataset
from train import BATCH_SIZE, model, train
from utils import cal_generate_num, print_generate_num, generating_dataset
from prediction import prediction

import numpy as np

from tensorflow.keras.models import load_model

NUM_CLASSES = 5

TRAIN_DIR = "./data_test/train/"
TEST_DIR = "./data_test/test/"

Vasculitis_generator_PATH = "./model_save/GAN_Vasculitis.h5"
Urticaria_generator_PATH = "./model_save/GAN_Urticaria.h5"

ORIGINAL_MODEL_PATH = "./model_save/resnet50_original.h5"
TRANSFORMED_MODEL_PATH = "./model_save/resnet50_transformed.h5"
GAN_MODEL_PATH = "./model_save/resnet50_gan.h5"

ORIGINAL_CONMAT_PATH = "./image/conmat_original.png"
TRANSFORMED_CONMAT_PATH = "./image/conmat_transformed.png"
GAN_CONMAT_PATH = "./image/conmat_gan.png"


### === Unaugmented Dataset ===
# origin=True: Don't do any data augmentation, just keep it origin
print("\n=== Original dataset Info ===")
train_data, train_labels, class_names = dataset(TRAIN_DIR, NUM_CLASSES, origin=True)

# print the test dataset Info
print("\n=== Test dataset Info ===")
test_data, test_labels, class_names = dataset(TEST_DIR, NUM_CLASSES, origin=True)

resnet50_model_0 = model()
train(resnet50_model_0, train_data, train_labels, save_path=ORIGINAL_MODEL_PATH)
resnet50_model_0 = load_model(ORIGINAL_MODEL_PATH)
prediction(resnet50_model_0, test_data, test_labels, class_names, save_path=ORIGINAL_CONMAT_PATH)


### === Transformed Dataset ===
# Get the train dataset
print("\n=== Transformed dataset Info ===")
train_data, train_labels, class_names = dataset(TRAIN_DIR, NUM_CLASSES)

# Train on the transformed dataset
resnet50_model_1 = model()
train(resnet50_model_1, train_data, train_labels, save_path=TRANSFORMED_MODEL_PATH)

# The prediction
resnet50_model_1 = load_model(TRANSFORMED_MODEL_PATH)
prediction(resnet50_model_1, test_data, test_labels, class_names, save_path=TRANSFORMED_CONMAT_PATH)


### === Augmented Dataset ===
# calculate the number we need to generate
generate_num, class_names = cal_generate_num(TRAIN_DIR, NUM_CLASSES)
print_generate_num(TRAIN_DIR, class_names, generate_num)

# get augmented dataset
generators = []
vasculitis_generator = load_model(Vasculitis_generator_PATH)
urticaria_generator = load_model(Urticaria_generator_PATH)
generators.append(vasculitis_generator)
generators.append(vasculitis_generator)
generators.append(vasculitis_generator)
generators.append(urticaria_generator)
generators.append(vasculitis_generator)

generated_data, generated_labels = generating_dataset(generators, generate_num)
print("\n=== GAN augmented dataset Info ===")
data, labels, class_names = dataset(TRAIN_DIR, NUM_CLASSES, 
                                    augment=True, aug_data=generated_data, aug_labels=generated_labels)

# train it!
resnet50_model_2 = model()
train(resnet50_model_2, data, labels, save_path=GAN_MODEL_PATH, augment=True)

# do the prediction
resnet50_model_2 = load_model(GAN_MODEL_PATH)
prediction(resnet50_model_2, test_data, test_labels, class_names, save_path=GAN_CONMAT_PATH, augment=True)

print("=== main.py finished ===\n")
