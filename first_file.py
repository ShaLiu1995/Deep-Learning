"""This is a test file"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def run_example():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print('Training image data: ', mnist.train.images.shape)
    print('Testing image data: ', mnist.test.images.shape)
    print('28 x 28 = ', 28 * 28)
    print('\nTrain image 1 is labelled one-hot as {0}'.format(mnist.train.labels[0, :]))
    image = np.reshape(mnist.train.images[0, :], [28, 28])
    plt.imshow(image, cmap='gray')
    # plt.show()

    current_data = mnist.train.next_batch(10)

    # Example image
    print('\nTrain image 1 is labelled one-hot as {0}'.format(current_data[1]))
    image = np.reshape(current_data[0], [28, 28])
    plt.imshow(image, cmap='gray')
    # plt.show()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # r = np.asarray([0, 0, 1])
    # loss = np.asarray([0, 0, 3]).T
    # print(np.matmul(r, loss))
    # print(loss)

    run_example()