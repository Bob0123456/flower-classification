import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def remove_noise(images):
    # Gaussian
    no_noise = []
    for i in range (len(images)):
        blured = cv2.GaussianBlur(images[i], (5, 5), 0)
        no_noise.append(blured)
    return no_noise


def preprocess():
    data = "flowers/"
    img_size = 128

    folders = os.listdir(data)
    print(folders)
    images = []
    labels = []
    for i in range(0, len(folders)):
        path = os.path.join(data,folders[i])
        for img in os.listdir(path):
            label = folders[i]
            img_path = os.path.join(path, img)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img_size, img_size))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # convert to correct colors
            images.append(np.array(img))
            labels.append(str(label))

    print(len(images))

    images = remove_noise(images)

    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(5, 5)
    for i in range(2):
        for j in range(2):
            l = rn.randint(0, len(images))
            ax[i, j].imshow(images[l])

    plt.tight_layout()
    plt.show()



    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    images = np.array(images)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = labels.reshape(len(labels), 1)
    Y = onehot_encoder.fit_transform(integer_encoded)
    X = np.array(images)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=100)
    return x_train, x_test, y_train, y_test

# x_train, x_test, y_train, y_test = preprocess()




