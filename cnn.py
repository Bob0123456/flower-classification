from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import open
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from keras.optimizers import Adam, SGD, Adagrad, Adadelta


def print_confusion_matrix(probs, labels, classes):
    accuracy_score(labels, probs)
    preds = np.argmax(probs, axis=1)
    probs = probs[:, 1]
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, classes, figsize=(10, 10))


x_train, x_test, y_train, y_test = open.preprocess()

# define the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(128, 128, 3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=96, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(5, activation="softmax"))  # the output layer
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=15,
          batch_size=256)


pred = model.predict(x_test)
score = model.evaluate(x_test, y_test, batch_size=256)
print(score)

model.summary()


