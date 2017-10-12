import tensorflow as tf
import numpy as np
import caffe
from caffe.proto import caffe_pb2
import util 
from nets import resnet_v2, resnet_utils
slim = tf.contrib.slim
caffemodel_path=util.io.get_absolute_path(
    '~/models/pvanet/PVA9.1_ImgNet_COCO_VOC0712plus_compressed.caffemodel')

class CaffeScope():
    def __init__(self):
        print('Loading Caffe file:', caffemodel_path)
        caffemodel_params = caffe_pb2.NetParameter()
        caffemodel_str = open(caffemodel_path, 'rb').read()
        caffemodel_params.ParseFromString(caffemodel_str)
        param_layers = ['Scale', 'Convolution', 'BatchNorm']
        def get_layers(layer_type):
            return [l for l in caffemodel_params.layer if l.type == layer_type]
        
        self.conv_layers = get_layers('Convolution')
        self.bn_layers = get_layers('BatchNorm')
        self.scale_layers = get_layers('Scale')
        
        def cvt(blob):
            shape = np.array(blob.shape.dim)
            data = np.array(blob.data)
            data = np.reshape(data, shape)
            return data
            
        self.conv_weights = [cvt(layer.blobs[0]) for layer in self.conv_layers]
        self.bn_moving_mean = [cvt(layer.blobs[0]) for layer in self.bn_layers]
        self.bn_moving_variance = [cvt(layer.blobs[1]) for layer in self.bn_layers]
        self.scale_gamma = [cvt(layer.blobs[0]) for layer in self.scale_layers]
        self.scale_beta = [cvt(layer.blobs[1]) for layer in self.scale_layers]
        
        self.conv_count = 0;
        self.moving_mean_count = 0;
        self.moving_variance_count = 0;
        self.gamma_count = 0;
        self.beta_count = 0;

    def conv_weights_init(self):
        def _initializer(shape, dtype, partition_info=None):
            # Weights: reshape and transpose dimensions.
            layer = self.conv_layers[self.conv_count]
            w = self.conv_weights[self.conv_count]
            w = np.transpose(w, (2, 3, 1, 0))
            np.testing.assert_equal(w.shape, shape)
            self.conv_count += 1;
            print('Load weights from convolution layer:', layer.name, w.shape, shape)
            return tf.cast(w, dtype)
        return _initializer
    def bn_init(self):
        def moving_average_initializer(shape, dtype, partition_info=None):
            moving_mean = self.bn_moving_mean[self.moving_mean_count]
            layer = self.bn_layers[self.moving_mean_count]
            np.testing.assert_equal(moving_mean.shape, shape)
            self.moving_mean_count += 1
            print('Load moving_mean params from BatchNorm layer:', layer.name)
            return moving_mean


        def moving_variance_initializer(shape, dtype, partition_info=None):
            layer = self.bn_layers[self.moving_variance_count]
            print('Load moving_variance params from BatchNorm layer:', layer.name)
            moving_variance = self.bn_moving_variance[self.moving_variance_count]
            np.testing.assert_equal(moving_variance.shape, shape)
            self.moving_variance_count += 1
            return moving_variance


        def gamma_initializer(shape, dtype, partition_info=None):
            layer = self.scale_layers[self.gamma_count]
            gamma = self.scale_gamma[self.gamma_count]
            np.testing.assert_equal(gamma.shape, shape)
            self.gamma_count += 1
            print('Load gamma params from Scale layer:', layer.name)
            return gamma

        def beta_initializer(shape, dtype, partition_info=None):
            beta = self.scale_beta[self.beta_count]
            layer = self.scale_layers[self.beta_count]
            np.testing.assert_equal(beta.shape, shape)
            self.beta_count += 1
            print('Load beta params from Scale layer:', layer.name)
            return beta

        params = {
            'moving_mean': moving_average_initializer,
            'moving_variance':moving_variance_initializer,
            'gamma':gamma_initializer,
            'beta': beta_initializer
        }
        return params


caffe_scope = CaffeScope()

fake_image = tf.placeholder(dtype = tf.float32, shape = [2, 512, 512, 3])

with slim.arg_scope(resnet_utils.resnet_arg_scope(
    batch_norm_param_initializer = caffe_scope.bn_init(),
    weights_initializer = caffe_scope.conv_weights_init()
    )):
    _, end_points = resnet_v2.resnet_v2_152(
              fake_image,
              is_training=True,
              reuse=False,
              scope='')
    for name in end_points:
        print name, end_points[name].shape
with tf.Session() as session:
    # Run the init operation.
    session.run(tf.global_variables_initializer())

    # Save model in checkpoint.
    saver = tf.train.Saver(write_version=2)
    parent_dir = util.io.get_dir(caffemodel_path)
    filename = util.io.get_filename(caffemodel_path)
    parent_dir = util.io.mkdir(util.io.join_path(parent_dir, 'pixel_link_pretrain'))
    filename = filename.replace('.caffemodel', '.ckpt')
    ckpt_path = util.io.join_path(parent_dir, filename)
    saver.save(session, ckpt_path, write_meta_graph=False)
