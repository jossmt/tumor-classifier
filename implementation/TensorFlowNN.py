import datetime

from nltk.app.nemo_app import colors
from tensorflow import keras
from matplotlib import pyplot as plt
import keras_tuner as kt
import tensorflow as tf
import PIL
import pathlib
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.applications.densenet import layers
import pandas as pd
from tensorflow.python.keras.layers import Conv2D

BINARY_METRICS = [
    'accuracy',
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]

BINARY_LOSS = tf.keras.losses.BinaryCrossentropy(from_logits=True),

MULTICLASS_METRICS = [
    'accuracy'
]

MULTICLASS_LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# TODO: 1. Add dropout/regularisation to reduce overfitting
# 2. Normalize data -- done
# 3. Resolve imbalances (highlight the distribution) using cost function or keras class weights
# 4. Add more metrics to the model.compile, use history to generate confusion matrix
# 5.

def tensor_nn_predict():
    train_ds, val_ds = load_multiclass_data()
    class_names = train_ds.class_names

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    Conv2D(16, 3, padding='same', activation='relu'),
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(512, 512, 1)),
        layers.Conv2D(16, 5, padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D((3,3)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=MULTICLASS_LOSS,
                  metrics=MULTICLASS_METRICS)

    model.summary()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    epochs = 20
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[tensorboard_callback],
        class_weight=class_weights_multiclass(class_names)
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    plot_metrics(history)

    model.predict()


# def tune_model(train_ds):
#     tuner = kt.Hyperband(model_builder,
#                          objective='val_accuracy',
#                          max_epochs=10,
#                          factor=3,
#                          directory='my_dir',
#                          project_name='intro_to_kt')
#
#     stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
#
#     tuner.search(train_ds, epochs=20, validation_split=0.2, callbacks=[stop_early])
#
#     # Get the optimal hyperparameters
#     best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
#
#     return tuner.hypermodel.build(best_hps)


# def model_builder(hp):
#
#     hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
#     model = Sequential([
#         layers.Rescaling(1. / 255, input_shape=(512, 512, 3)),
#         layers.Normalization(axis=None),
#         layers.Conv2D(16, 3, padding='same', activation='relu'),
#         layers.MaxPooling2D(),
#         layers.Dropout(0.2),
#         layers.Conv2D(32, 3, padding='same', activation='relu'),
#         layers.MaxPooling2D(),
#         layers.Conv2D(64, 3, padding='same', activation='relu'),
#         layers.MaxPooling2D(),
#         layers.Flatten(),
#         layers.Dense(units=hp_units, activation='relu'),
#         layers.Dropout(0.2),
#         layers.Dense(units=hp_units, activation='relu'),
#         layers.Dense(4, activation='softmax')
#     ])
#     hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
#
#     model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
#                   loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                   metrics=['accuracy'])
#
#     return model


def load_binary_data():
    data_dir = pathlib.Path('../dataset/binary_tf')

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print('total images')
    print(image_count)

    no_tumor_length = len(list(data_dir.glob('no_tumor/*')))
    print(no_tumor_length)

    tumor_length = len(list(data_dir.glob('tumor/*')))
    print(tumor_length)

    batch_size = 32
    img_height = 512
    img_width = 512

    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        color_mode='grayscale',
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        color_mode='grayscale',
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         print('should plot')
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    #
    # plt.show()
    return train_ds, val_ds


def load_multiclass_data():
    data_dir = pathlib.Path('../dataset/multiclass_tf')

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print('total images')
    print(image_count)

    no_tumor_length = len(list(data_dir.glob('no_tumor/*')))
    print(no_tumor_length)

    tumor_length = len(list(data_dir.glob('tumor/*')))
    print(tumor_length)

    batch_size = 32
    img_height = 512
    img_width = 512

    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        color_mode='grayscale',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        color_mode='grayscale',
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         print('should plot')
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    #
    # plt.show()
    return train_ds, val_ds


def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()

    plt.show()


def class_weights_multiclass(class_names):
    df = pd.read_csv('../dataset/label.csv')
    counts = df['label'].value_counts().to_dict()
    print(class_names)
    return {
        class_names.index('no_tumor'): counts.get('no_tumor'),
        class_names.index('glioma_tumor'): counts.get('glioma_tumor'),
        class_names.index('meningioma_tumor'): counts.get('meningioma_tumor'),
        class_names.index('pituitary_tumor'): counts.get('pituitary_tumor'),
    }

def class_weights_binary(class_names):
    df = pd.read_csv('../dataset/label.csv')
    counts = df['label'].value_counts().to_dict()
    return {class_names.index('no_tumor'): counts.get('no_tumor'),
            1 - class_names.index('no_tumor'): df.shape[0] - counts.get('no_tumor')}
