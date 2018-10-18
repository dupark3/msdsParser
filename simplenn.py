# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(x_train.shape)
print(len(x_train))
print(y_train)
print(x_test.shape)
print(len(x_test))
print(y_test)

x_train = x_train / 255.0
x_test = x_test / 255.0

# See sample images
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
#     plt.xlabel("Digit : " + str(y_train[i]))
# plt.show()

def create_model():
    model = keras.models.Sequential([
      keras.layers.Flatten(input_shape=(28,28)),
      keras.layers.Dense(512, activation=tf.nn.relu),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model()
model.summary()
model.fit(x_train, y_train, epochs=10)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Accuracy: ', test_acc)

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save('model.h5')