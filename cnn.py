from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import open
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
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

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.2,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

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


model.fit_generator(datagen.flow(x_train,y_train, batch_size=256),
          epochs=15,
          validation_data = (x_test, y_test))


pred = model.predict(x_test)
score = model.evaluate(x_test, y_test, batch_size=256)
print(score)
model.summary()




