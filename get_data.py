from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

def get_images(mypath, size = 256, channels = 1):

    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    onlyfiles.sort()
    images = np.empty(len(onlyfiles), dtype=object)
    images = np.zeros((len(onlyfiles), size, size))

    for n in range(len(onlyfiles)):
        images[n] = cv2.imread( join(mypath,onlyfiles[n]), 0)
        #images[n] = cv2.resize(temp, (size, size), interpolation = cv2.INTER_AREA)

    #print(len(images))

    return images

def load_data():

    training_images_path = "/train_images"
    training_labels_path = "/train_masks"
    dev_images_path = "/dev_images"
    dev_labels_path = "/dev_masks"

    training_images = get_images(training_images_path)
    training_masks = get_images(training_labels_path)
    dev_images = get_images(validation_images_path)
    dev_masks = get_images(validation_labels_path)

    training_images = training_images.reshape(training_images.shape[0], 256, 256, 1)
    dev_images = validation_images.reshape(dev_images.shape[0], 256, 256, 1)
    training_masks = training_masks.reshape(training_masks.shape[0], 256, 256, 1)
    dev_masks = validation_masks.reshape(dev_masks.shape[0], 256, 256, 1)

    return training_images, training_masks, dev_images, dev_masks

def show_predictions():

    t = 256
    x = 60
    n = 10

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # MRI
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i+x].reshape(t, t))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Ground Truth
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(y_test[i+x].reshape(t, t))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Prediction
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(decoded_imgs[i+x].reshape(t, t))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('Results.png')
    plt.show()
