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
    plt.show()

    current_data = mnist.train.next_batch(10)

    # Example image
    print('\nTrain image 1 is labelled one-hot as {0}'.format(current_data[1]))
    image = np.reshape(current_data[0], [28, 28])
    plt.imshow(image, cmap='gray')
    plt.show()


def run_train(minibatch_size=500, max_iterations=200, step_size=0.1):
    # minibatch_size = 500
    data_minibatch = mnist.train.next_batch(minibatch_size)

    # max_iterations = 200  # choose the max number of iterations
    # step_size = 0.1  # choose your step size
    W = np.random.rand(10, 784)  # choose your starting parameters (connection weights)
    b = np.random.rand(10, )  # choose your starting parameters (biases)
    training_loss_history = []
    for iter in range(0, max_iterations):
        # current_data=mnist.train.next_batch(1)
        # note you need to change this to your preferred data format.
        current_parameters = [W, b]
        W_grad, b_grad = lr_gradient(current_parameters, data_minibatch)
        training_loss_history.append(lr_loss(current_parameters, data_minibatch))
        W = W - step_size * W_grad
        b = b - step_size * b_grad
    return W, b


def lr_gradient(current_parameters, data_minibatch):
    # calculate the gradient on the data
    W_grad = np.zeros((10, 784))
    b_grad = np.zeros((10, ))

    W = current_parameters[0]
    b = current_parameters[1]
    x = data_minibatch[0]
    r = data_minibatch[1]
    N = x.shape[0]
    for i in range(N):
        gamma = np.matmul(W, x[i]) + b
        l_grad_gamma = softmax(gamma) - r[i]
        W_grad += np.matmul(l_grad_gamma.reshape(1, -1).T, x[i].reshape(1, -1))
        b_grad += l_grad_gamma
    W_grad /= 1.0 * N
    b_grad /= 1.0 * N
    return W_grad, b_grad


def lr_loss(current_parameters, data_minibatch):
    # calculate the loss
    W = current_parameters[0]
    b = current_parameters[1]
    x = data_minibatch[0]
    r = data_minibatch[1]
    N = x.shape[0]
    avg_loss = 0.0
    for i in range(N):
        gamma = np.matmul(W, x[i]) + b
        curr_loss = np.matmul(r[i], np.log(softmax(gamma)))
        avg_loss -= curr_loss
    return avg_loss / (1.0 * N)


def softmax(gamma):
    """Compute softmax values"""
    numerator = np.exp(gamma)
    denominator = sum(numerator)
    return numerator / (1.0 * denominator)


def run_test(W, b):
    data_minibatch = mnist.test.next_batch(1000)
    x = data_minibatch[0]
    r = data_minibatch[1]
    N = x.shape[0]
    accuracy = 0
    for i in range(N):
        prediction = softmax(np.matmul(W, x[i]) + b)
        if np.argmax(prediction) == np.argmax(r[i]):
            accuracy += 1
    accuracy /= (1.0 * N)
    return accuracy


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    history = []
    for minibatch_size in [500, 1000]:
        for max_iterations in [500, 1000]:
            for step_size in [4, 2, 1, 0.5]:
                W_train, b_train = run_train(minibatch_size, max_iterations, step_size)
                acc = run_test(W_train, b_train)
                print(acc, minibatch_size, max_iterations, step_size)
                history.append([acc, minibatch_size, max_iterations, step_size])

    list.sort(history, key=lambda h: h[0], reverse=True)
    print('Best config: {}'.format(history[:5]))

