from dataset import dataset, print_class_info

import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam


### User define
TRAIN_DIR = "./Dermnet/"
NUM_CLASSES = 5
IMAGE_SIZE = 128
BATCH_SIZE = 64
NUM_EPOCHS = 20


def model():
    resnet50 = ResNet50(
        include_top = False,
        weights = "imagenet",
        input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3),
        pooling = 'avg',
        classes = NUM_CLASSES,
    )
    prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='relu', bias_regularizer=regularizers.l2(1e-4))

    model = Sequential([
        resnet50,
        prediction_layer,
        layers.Flatten(),
        layers.BatchNormalization(),
        layers.Activation('softmax'),
    ])

    return model

def train(model, X, y, save_path, augment=False):
    # 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
    model.compile(optimizer=Adam(lr=1e-5),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # training
    model.fit(X, y, 
              validation_split=0.1,
              epochs=NUM_EPOCHS,
              batch_size=BATCH_SIZE,
              verbose=1)

    model.save(save_path)
    print("Save the modet to {}".format(save_path))


if __name__ == "__main__":
    data, labels, class_names = dataset(TRAIN_DIR, NUM_CLASSES)

    # Print the class labels and the corresponding names
    print_class_info(class_names, labels)

    # resnet50_model = model(data_augmentation)
    resnet50_model = model()

    train(resnet50_model, data, labels)
    print("=== Training has done! ===\n")
