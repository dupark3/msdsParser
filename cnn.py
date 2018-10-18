# convolusional neural networks

import pandas as py
import numpy as np

#image tools
from matplotlib import pyplot as plt
from PIL import Image

#filesystem
import os
from glob import glob
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

mnist.train.images.reshape(-1, 28, 28, 1).shape

with tf.device('/gpu:0'):
    graph = tf.Graph()
    with graph.as_default():
        batch_size=128
        num_hidden=1024
        keep_prob=0.5
        image_size=28
        num_labels=10
        patch_size=5
        num_channels=1
        depth=16

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        #input data
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(mnist.validation.images.reshape(-1, 28, 28, 1))
        tf_test_dataset = tf.constant(mnist.test.images.reshape(-1, 28, 28, 1))
        keep_prob = tf.placeholder(tf.float32)

        W_conv1 = weight_variable([5,5,1,32])
        b_conv1 = bias_variable([32])
        W_conv2 = weight_variable([5,5,32,64])
        b_conv2 = bias_variable([64])
        W_fc1 = weight_variable([7*7*64,1024])
        b_fc1 = bias_variable([1024])
        W_fc2 = weight_variable([1024,10])
        b_fc2 = bias_variable([10])

        def model(data):
            h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)
            h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
            return tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        logits = model(tf_train_dataset)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
        test_prediction = tf.nn.softmax(model(tf_test_dataset))
def accuracy(predictions,labels):
    return (100.0 * np.sum(np.argmax(predictions,1) == np.argmax(labels,1)) / predictions.shape[0])

num_steps = 8001

with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        batch_data, batch_labels = mnist.train.next_batch(batch_size)
        batch_data = batch_data.reshape(-1, 28, 28, 1)
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : 0.5}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if(step%500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" %accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(feed_dict={keep_prob:1}), mnist.validation.labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(feed_dict={keep_prob:1}), mnist.test.labels))

    saver = tf.train.Saver()
    saver.save(session, 'model3')
