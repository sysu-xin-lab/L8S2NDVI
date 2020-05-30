import tensorflow as tf


def conv2d(net,
           filters,
           kernel_size=3,
           strides=1,
           activation=tf.nn.relu,
           padding='same',
           name='conv'):
    return tf.layers.conv2d(net,
                            filters, (3, 3),
                            strides=strides,
                            activation=activation,
                            padding='same',
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                            name=name)


def srcnn(net, numOutChannels):
    net = conv2d(net, 128, 9, name='conv9x9')
    net = conv2d(net, 64, 1, name='conv1x1')
    net = conv2d(net, numOutChannels, 5, activation=None, name='conv5x5')
    return net
