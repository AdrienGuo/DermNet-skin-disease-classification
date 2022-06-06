from train import BATCH_SIZE
from dataset import dataset, print_class_info

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report


TEST_DIR = "./Dermnet/"
NUM_CLASSES = 5
IMAGE_DIR = "./image/"
MODEL_PATH = "./model_save/resnet50_wo_gan.h5"


def prediction(model, data, labels, class_names, save_path):
    preds = model.predict(data, batch_size=BATCH_SIZE)
    y_pred = np.argmax(preds, axis=-1)
    
    con_mat = tf.math.confusion_matrix(labels=labels, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm,
                        index = class_names,
                        columns = class_names)

    figure = plt.figure(figsize=(10, 10))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.savefig(save_path)
    print("Save the prediction result to ", save_path)

    print("=== Other results ===")
    print(classification_report(labels, y_pred))


if __name__ == "__main__":
    data, labels, class_names = dataset(TEST_DIR, NUM_CLASSES)
    
    # Print the class labels and the corresponding names
    print_class_info(class_names, labels)

    resnet50_model = load_model(MODEL_PATH)
    prediction(resnet50_model, data, labels, class_names)

    print("=== Prediction has done! ===\n")
