import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt


NUM_CLASSES = 10
INPUT_SHAPE = (227,227, 3)
CIFAR_CLASSES_PT = [
    "Avião",
    "Automóvel",
    "Pássaro",
    "Gato",
    "Cervo",
    "Cachorro",
    "Sapo",
    "Cavalo",
    "Navio",
    "Caminhão"
]


def build_alexnet(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', dtype='float32'))
    return model


def preprocess_image(image):
    image = tf.image.resize(image, (227, 227))
    return image / 255.0

def preprocess_data(image, label):
    image = preprocess_image(image)
    label = tf.squeeze(tf.one_hot(label, 10))
    return image, label


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=50000).batch(64).prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.map(preprocess_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(64).prefetch(tf.data.experimental.AUTOTUNE)

model = build_alexnet()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=50, validation_data=test_dataset)


x_validation = x_test[:10]
y_validation = y_test[:10]

validation_ds = tf.data.Dataset.from_tensor_slices(x_validation).map(preprocess_image).batch(10)
predictions = model.predict(validation_ds)

def show_predictions(predictions, x, y):
    index = np.random.randint(0, len(x_test))
    test_image = x_test[index]
    true_label = np.argmax(y_test[index])

    for i in range(6):
        true_label = y[i]
        test_image = x[i]
        plt.figure(figsize = (1,1))
        predicted_label = np.argmax(predictions[i])
        plt.imshow(test_image)
        plt.title(f'Categoria correta: {CIFAR_CLASSES_PT[true_label[0]]} , Categoria via ALEXNET: {CIFAR_CLASSES_PT[predicted_label]}')
        plt.show()

show_predictions(predictions, x_validation, y_validation)
