from tensorflow.keras.datasets import mnist
from keras import models
from matplotlib import pyplot
import matplotlib.pyplot as plt
from numpy import expand_dims
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar100, cifar10
(X_train, _), (X_test, _) = mnist.load_data()

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# Scale X to range between 0 and 1
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
import numpy as np

noise_factor = 0.5

noise = noise_factor * np.random.normal(loc= 0.5, scale= 0.5, size= X_train.shape)
X_train_noisy = X_train + noise
X_train_noisy = np.clip(X_train_noisy, 0., 1.)

noise = noise_factor * np.random.normal(loc= 0.5, scale= 0.5, size= X_test.shape)
X_test_noisy = X_test + noise
X_test_noisy = np.clip(X_test_noisy, 0., 1.)
from tensorflow.keras.models import model_from_json
import warnings; warnings.filterwarnings('ignore')

with open('/usr/autoencoder/logs/Encoder_model.json', 'r') as f:
    Myencoder = model_from_json(f.read())
Myencoder.load_weights("/usr/autoencoder/logs/Encoder_weights.h5")

with open('/usr/autoencoder/logs/Autoencoder_model.json', 'r') as f:
    MyAutoencoder = model_from_json(f.read())
MyAutoencoder.load_weights("/usr/autoencoder/logs/Autoencoder_weights.h5")
# Pick randomly some images from test set
num_images = 20
random_test_images = np.random.randint(X_test.shape[0], size=num_images)

# Predict the Encoder and the Autoencoder outputs from the noisy test images
encoded_imgs = Myencoder.predict(X_test_noisy)
decoded_imgs = MyAutoencoder.predict(X_test_noisy)
fig = plt.figure(figsize=(8, 4))
fig.tight_layout()
for i, image_idx in enumerate(random_test_images):
    # Plot original image
    ax = plt.subplot(4, num_images, i + 1)
    plt.imshow(X_test[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == num_images // 2:
        ax.set_title('Oryginalne Obrazy')

    # Plot noised image
    ax = plt.subplot(4, num_images, num_images + i + 1)
    plt.imshow(X_test_noisy[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == num_images // 2:
        ax.set_title('Zaszumione Obrazy')

    # Plot encoded image
    ax = plt.subplot(4, num_images, 2 * num_images + i + 1)
    plt.imshow(encoded_imgs[image_idx].reshape(7, 7))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == num_images // 2:
        ax.set_title('Wąskie gardło')

    # Plot reconstructed image
    ax = plt.subplot(4, num_images, 3 * num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == num_images // 2:
        ax.set_title('Odszumione Obrazy')
plt.show()
layer_outputs = [layer.output for layer in MyAutoencoder.layers]
activation_model = models.Model(inputs = MyAutoencoder.input, outputs = layer_outputs)
dimensions = [1,2,3,4,5,6,7,8,9,10,11]
for i in dimensions:
    img = expand_dims(X_test_noisy[6], axis=0)
    feature_maps = activation_model.predict(img)
    feature_maps = feature_maps[i];
    if i==5 or i==11:
        ax = pyplot.subplot(1, 1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(feature_maps[0, :, :, 0], cmap='gray')
        pyplot.show()
        filters, biases = MyAutoencoder.layers[i].get_weights()
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        # plot first few filters
        n_filters, ix = 1, 1
        for i in range(n_filters):
            # get the filter
            f = filters[:, :, :, i]
            # plot each channel separately
            for j in range(1):
                # specify subplot and turn of axis
                ax = pyplot.subplot(1, 1, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                pyplot.imshow(f[:, :, j], cmap='gray')
                ix += 1
        pyplot.show()
    elif i == 10:
        dim_1 = 4;
        dim_2 = 4;
        ix = 1
        for _ in range(dim_1):
            for _ in range(dim_2):
                    ax = pyplot.subplot(dim_1, dim_2, ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
                    ix += 1
        pyplot.show()
        n_filters, ix = 16, 1
        filters, biases = MyAutoencoder.layers[i].get_weights()
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        for i in range(n_filters):
            # get the filter
            f = filters[:, :, :, i]
            # plot each channel separately
            for j in range(1):
                # specify subplot and turn of axis
                ax = pyplot.subplot(4, 4, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                pyplot.imshow(f[:, :, j], cmap='gray')
                ix += 1
        pyplot.show()
    else:
        dim_1 = 4;
        dim_2 = 8;
        ix = 1
        for _ in range(dim_1):
            for _ in range(dim_2):
                    ax = pyplot.subplot(dim_1, dim_2, ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
                    ix += 1
        pyplot.show()
        if i==1 or i==3 or i==6 or i==8:
            filters, biases = MyAutoencoder.layers[i].get_weights()
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)
            # plot first few filters
            n_filters, ix = 32, 1
            for i in range(n_filters):
                # get the filter
                f = filters[:, :, :, i]
                # plot each channel separately
                for j in range(1):
                    # specify subplot and turn of axis
                    ax = pyplot.subplot(4, 8, ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # plot filter channel in grayscale
                    pyplot.imshow(f[:, :, j], cmap='gray')
                    ix += 1
            # show the figure

            pyplot.show()