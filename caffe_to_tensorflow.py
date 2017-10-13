import tensorflow as tf
import numpy as np
import caffe
from caffe.proto import caffe_pb2
import util 
slim = tf.contrib.slim
caffemodel_path=util.io.get_absolute_path(
    '~/models/pvanet/PVA9.1_ImgNet_COCO_VOC0712plus_compressed.caffemodel')

def get_layer_name(scope_name):
    if util.str.contains(scope_name, '/bn_scale'):
        scope_name = util.str.replace_all(scope_name,'/bn_scale' , '/scale')
        
    if util.str.contains(scope_name, '/incep/out/conv'):
        scope_name = util.str.replace_all(scope_name, '/incep/out/conv' , '/out/conv')

    if util.str.contains(scope_name, '/incep/proj'):
        scope_name = util.str.replace_all(scope_name, '/incep/proj' , '/proj')
        
    return scope_name

    
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
            
        self.conv_weights = {get_layer_name(layer.name): cvt(layer.blobs[0]) for layer in self.conv_layers}
        self.bn_moving_mean = {get_layer_name(layer.name): cvt(layer.blobs[0]) for layer in self.bn_layers}
        self.bn_moving_variance = {get_layer_name(layer.name): cvt(layer.blobs[1]) for layer in self.bn_layers}
        self.scale_gamma = {get_layer_name(layer.name): cvt(layer.blobs[0]) for layer in self.scale_layers}
        self.scale_beta = {get_layer_name(layer.name): cvt(layer.blobs[1]) for layer in self.scale_layers}
        
    def conv_weights_init(self):
        def _initializer(shape, dtype, partition_info=None):
            # Weights: reshape and transpose dimensions.
            layer_name = tf.get_variable_scope().name
            layer_name =  get_layer_name(layer_name)
            w = self.conv_weights[layer_name]
            w = np.transpose(w, (2, 3, 1, 0))
            print('Load weights from convolution layer:', layer_name, w.shape, shape)
            np.testing.assert_equal(shape, w.shape)
            return tf.cast(w, dtype)
        return _initializer
    
    def gamma_init(self):
        def _initializer(shape, dtype, partition_info=None):
            layer_name = tf.get_variable_scope().name
            gamma = self.scale_gamma[layer_name]
            print('Load gamma params from Scale layer:', layer_name)
            np.testing.assert_equal(shape, gamma.shape)
            return gamma
        return _initializer
    
    def beta_init(self):
        def _initializer(shape, dtype, partition_info=None):
            layer_name = tf.get_variable_scope().name
            beta = self.scale_beta[layer_name]
            print('Load beta params from Scale layer:', layer_name)
            np.testing.assert_equal(shape, beta.shape)
            return beta
        return _initializer
    
    def bn_init(self):
        def moving_average_initializer(shape, dtype, partition_info=None):
            layer_name = tf.get_variable_scope().name
            moving_mean = self.bn_moving_mean[layer_name]
            print('Load moving_mean params from BatchNorm layer:', layer_name)
            np.testing.assert_equal(shape, moving_mean.shape)
            return moving_mean


        def moving_variance_initializer(shape, dtype, partition_info=None):
            layer_name = tf.get_variable_scope().name
            moving_variance = self.bn_moving_variance[layer_name]
            print('Load moving_variance params from BatchNorm layer:', layer_name)
            np.testing.assert_equal(moving_variance.shape, shape)
            return moving_variance

        params = {
            'moving_mean': moving_average_initializer,
            'moving_variance':moving_variance_initializer
        }
        return params


caffe_scope = CaffeScope()

fake_image = tf.placeholder(dtype = tf.float32, shape = [2, 512, 512, 3])
from pvanet import pvanet, pvanet_scope

with slim.arg_scope(pvanet_scope(
    is_training = False,
    batch_norm_param_initializer = caffe_scope.bn_init(),
    weights_initializer = caffe_scope.conv_weights_init(),
    beta_initializer = caffe_scope.beta_init(),
    gamma_initializer = caffe_scope.gamma_init()
    )):
    _, end_points = net, end_points = pvanet(fake_image)
    for name in end_points:
        print name, end_points[name].shape
with tf.Session() as session:
    # Run the init operation.
    session.run(tf.global_variables_initializer())

    # Save model in checkpoint.
    saver = tf.train.Saver(write_version=2)
    parent_dir = util.io.get_dir(caffemodel_path)
    filename = util.io.get_filename(caffemodel_path)
    parent_dir = util.io.mkdir(util.io.join_path(parent_dir, 'pretrained-pva'))
    filename = filename.replace('.caffemodel', '.ckpt')
    ckpt_path = util.io.join_path(parent_dir, filename)
    saver.save(session, ckpt_path, write_meta_graph=False)
