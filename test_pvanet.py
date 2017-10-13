"""
Just to make sure the build process of pvanet works well
"""
from pvanet import pvanet, pvanet_scope
import util
import tensorflow as tf
slim = tf.contrib.slim

is_training = False
with slim.arg_scope(pvanet_scope(is_training)):
    inputs = tf.placeholder(dtype = tf.float32, shape = [None, 768, 1280, 3])
    net, end_points = pvanet(inputs)
    for k in sorted(end_points.keys()):
        print k, end_points[k].shape
    print net.shape