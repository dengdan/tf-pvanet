"""
Just to make sure the build process of pvanet works well
"""
from pvanet import pvanet, pvanet_scope

import tensorflow as tf
slim = tf.contrib.slim

is_training = False
with slim.arg_scope(pvanet_scope(is_training)):
    inputs = tf.placeholder(dtype = tf.float32, shape = [None, 640, 1056, 3])
    net, end_points = pvanet(inputs)
    vars = tf.model_variables()
    for var in vars:
        print var.name
    print len(vars)
    