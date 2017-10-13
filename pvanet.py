
"""
PVANet mainly consists of two different kinds of blocks: 
    1. conv-crelu-bn blocks
    2. inception blocks
"""

from collections import namedtuple
import tensorflow as tf
slim = tf.contrib.slim

BLOCK_TYPE_MCRELU = 'BLOCK_TYPE_MCRELU'
BLOCK_TYPE_INCEP = 'BLOCK_TYPE_INCEP' 

BlockConfig = namedtuple('BlockConfig', 
              'stride, num_outputs, preact_bn, block_type')


def __conv(net, kernel_size, stride, num_outputs, scope = 'conv'):
    net = slim.conv2d(inputs = net, 
                  num_outputs = num_outputs, 
                  kernel_size = kernel_size,
                  activation_fn = None,
                  stride = stride,
                  scope = scope
                  )
    return net

@slim.add_arg_scope
def __scale(inputs, beta_initializer, gamma_initializer, is_training, scope = None):
    if scope is not None:
        scope = '%s_scale'%(scope)
    else:
        scope = 'scale'
    with tf.variable_scope(scope):
        dtype = inputs.dtype.base_dtype
        input_shape = inputs.get_shape()
        param_shape = input_shape[-1:]
        beta = tf.get_variable(name = 'beta', 
                               shape = param_shape,
                               dtype = dtype, 
                               initializer = beta_initializer, 
                               trainable = is_training
                               )
        gamma = tf.get_variable(name = 'gamma', 
                       shape = param_shape,
                       dtype = dtype, 
                       initializer = gamma_initializer, 
                       trainable = is_training
                       )
        
        return inputs * gamma + beta
    
def __bn_relu_conv(net, kernel_size, stride,num_outputs, scope = ''):
    with tf.variable_scope(scope):
        net = slim.batch_norm(net, scope = 'bn')
        net = __scale(net)
        net = tf.nn.relu(net, name = 'relu')
        net = __conv(net, kernel_size, stride, num_outputs)
        return net
    
def __conv_bn_relu(net, kernel_size, stride,num_outputs, scope = ''):
    with tf.variable_scope(scope):
        net = __conv(net, kernel_size, stride, num_outputs)
        net = slim.batch_norm(net, scope = 'bn')
        net = __scale(net)
        net = tf.nn.relu(net, name = 'relu')
        return net
def __bn_crelu(net):
        net = slim.batch_norm(net, scope = 'bn')
        # negation of bn results
        with tf.name_scope('neg'):
            neg_net = -net
        
        # concat bn and neg-bn
        with tf.name_scope('concat'):
            net = tf.concat([net, neg_net], axis = -1)
        net = __scale(net)
        # relu
        net = tf.nn.relu(net, name = 'relu')
        return net
    

def __conv_bn_crelu(net, kernel_size, stride, num_outputs, scope = ''):
    with tf.variable_scope(scope):
        net = __conv(net, kernel_size, stride, num_outputs)
        return __bn_crelu(net)
    
def __bn_crelu_conv(net, kernel_size, stride, num_outputs, scope = ''):
    with tf.variable_scope(scope):
        net = __bn_crelu(net)
        return __conv(net, kernel_size, stride, num_outputs)
    
def __mCReLU(inputs, mc_config):
    """
    every cReLU has at least three conv steps:
        conv_bn_relu, conv_bn_crelu, conv_bn_relu
    if the inputs has a different number of channels as crelu output,
    an extra 1x1 conv is added before sum.
    """
#     print tf.get_variable_scope().name
#     import pdb
#     pdb.set_trace()

    if mc_config.preact_bn:
        conv1_fn = __bn_relu_conv
        conv1_scope = '1'
    else:
        conv1_fn = __conv
        conv1_scope = '1/conv'
    
    sub_conv1 = conv1_fn(inputs, 
                            kernel_size = 1, 
                            stride = mc_config.stride, 
                            num_outputs = mc_config.num_outputs[0], 
                            scope = conv1_scope)
    
    sub_conv2 = __bn_relu_conv(sub_conv1, 
                            kernel_size = 3, 
                            stride = 1, 
                            num_outputs = mc_config.num_outputs[1], 
                            scope = '2')

    sub_conv3 = __bn_crelu_conv(sub_conv2, 
                           kernel_size = 1, 
                           stride = 1, 
                           num_outputs = mc_config.num_outputs[2], 
                           scope = '3')
    
    if inputs.shape.as_list()[-1] == mc_config.num_outputs[2]:
        conv_proj = inputs
    else:
        conv_proj = __conv(inputs, 
                        kernel_size = 1, 
                        stride = mc_config.stride, 
                        num_outputs = mc_config.num_outputs[2], 
                        scope = 'proj')

    conv = sub_conv3 + conv_proj
    
    return conv



def __inception_block(inputs, block_config):
    num_outputs = block_config.num_outputs.split() # e.g. 64 24-48-48 128
    stride = block_config.stride
    num_outputs = [s.split('-') for s in num_outputs]
    inception_outputs = int(num_outputs[-1][0])
    num_outputs = num_outputs[:-1]
    pool_path_outputs = None
    if stride > 1:
        pool_path_outputs = num_outputs[-1][0]
        num_outputs = num_outputs[:-1]
    
    scopes = [['0']] # follow the name style of caffe pva
    kernel_sizes = [[1]]
    for path_idx, path_outputs in enumerate(num_outputs[1:]):
        path_idx += 1
        path_scopes = ['{}_reduce'.format(path_idx)]
        path_scopes.extend(['{}_{}'.format(path_idx, i - 1) 
                            for i in range(1, len(path_outputs))])
        scopes.append(path_scopes)
        
        path_kernel_sizes = [1, 3, 3][:len(path_outputs)]
        kernel_sizes.append(path_kernel_sizes)

    paths = []
    if block_config.preact_bn:
        preact = slim.batch_norm(inputs, scope = 'bn')
        preact = __scale(preact)
        preact = tf.nn.relu(preact, name = 'relu')
    else:
        preact = inputs
        
    path_params = zip(num_outputs, scopes, kernel_sizes)
    for path_idx, path_param in enumerate(path_params):
        path_net = preact
        for conv_idx, (num_output, scope, kernel_size) in \
                    enumerate(zip(*path_param)):
            if conv_idx == 0:
                conv_stride = stride
            else:
                conv_stride = 1
            path_net = __conv_bn_relu(path_net, kernel_size, 
                                      conv_stride, num_output, scope)
        paths.append(path_net)    
    
    if stride > 1:
        path_net = slim.pool(inputs, kernel_size = 3, padding='SAME',
                             stride = 2, scope = 'pool')
        path_net = __conv_bn_relu(path_net, 
                                  kernel_size = 1, 
                                  stride = 1, 
                                  num_outputs = pool_path_outputs, 
                                  scope = 'poolproj')
        paths.append(path_net)
    block_net = tf.concat(paths, axis = -1)
    block_net = __conv(block_net, 
                       kernel_size = 1, 
                       stride =1, 
                       num_outputs = inception_outputs, 
                       scope = 'out/conv')
    
    if inputs.shape.as_list()[-1] == inception_outputs:
        proj = inputs
    else:
        proj = __conv(inputs, 
                      kernel_size = 1, 
                      stride = stride, 
                      num_outputs = inception_outputs, 
                      scope = 'proj')
    
    return block_net + proj
        
def __conv_stage(inputs, block_configs, scope, end_points):
    net = inputs
    for idx, bc in enumerate(block_configs):
        if bc.block_type == BLOCK_TYPE_MCRELU:
            block_scope = '{}_{}'.format(scope, idx + 1)
            fn = __mCReLU
        elif bc.block_type == BLOCK_TYPE_INCEP:
            block_scope = '{}_{}/incep'.format(scope, idx + 1)
            fn = __inception_block
        with tf.variable_scope(block_scope):
            net = fn(net, bc)
            end_points[block_scope] = net
    end_points[scope] = net
    return net
        
        
def pvanet_scope(is_training, 
                 weights_initializer = slim.xavier_initializer(), 
                 batch_norm_param_initializer = None,
                 beta_initializer = tf.zeros_initializer(), 
                 gamma_initializer = tf.ones_initializer(),
                 weight_decay = 0.99):
    l2_regularizer = slim.l2_regularizer(weight_decay)
    with slim.arg_scope([slim.conv2d], 
                        padding = 'SAME',
                        weights_initializer = weights_initializer,
                        weights_regularizer = l2_regularizer
                        ):
        with slim.arg_scope([slim.batch_norm], 
                            is_training = is_training,
                            scale = False, 
                            center = False,
                            param_initializers = batch_norm_param_initializer
                            ):
            with slim.arg_scope([__scale], 
                                is_training = is_training, 
                                beta_initializer = beta_initializer, 
                                gamma_initializer = gamma_initializer):
                with slim.arg_scope([slim.pool], 
                                    pooling_type = 'MAX', 
                                    padding='SAME') as sc:
                    return sc
                
def pvanet(net, include_last_bn_relu = True):
    end_points = {}
    
    # conv stage 1
    # /2, for the stride of 2
    conv1_1 = __conv_bn_crelu(net, 
                                kernel_size = (7, 7), 
                                stride = 2, 
                                num_outputs = 16, 
                                scope = 'conv1_1')
    # /2
    pool1 = slim.pool(conv1_1, kernel_size = 3, stride = 2, scope = 'pool1')
    
    
    conv2 = __conv_stage(pool1, 
         block_configs = [
            BlockConfig(1, (24, 24, 64), False, BLOCK_TYPE_MCRELU),
            BlockConfig(1, (24, 24, 64), True, BLOCK_TYPE_MCRELU),
            BlockConfig(1, (24, 24, 64), True, BLOCK_TYPE_MCRELU)], 
         scope = 'conv2',
         end_points = end_points)
    
    conv3 = __conv_stage(conv2,
         block_configs = [
            BlockConfig(2, (48, 48, 128), True, BLOCK_TYPE_MCRELU),
            BlockConfig(1, (48, 48, 128), True, BLOCK_TYPE_MCRELU),
            BlockConfig(1, (48, 48, 128), True, BLOCK_TYPE_MCRELU),
            BlockConfig(1, (48, 48, 128), True, BLOCK_TYPE_MCRELU)],
         scope = 'conv3',
         end_points = end_points)
    
    
    conv4 = __conv_stage(conv3, 
        block_configs = [
            BlockConfig(2, '64 48-128 24-48-48 128 256', True, BLOCK_TYPE_INCEP),
            BlockConfig(1, '64 64-128 24-48-48 256', True, BLOCK_TYPE_INCEP),
            BlockConfig(1, '64 64-128 24-48-48 256', True, BLOCK_TYPE_INCEP),
            BlockConfig(1, '64 64-128 24-48-48 256', True, BLOCK_TYPE_INCEP)],
        scope = 'conv4',
        end_points = end_points)
    
    conv5 = __conv_stage(conv4, 
        block_configs = [
            BlockConfig(2, '64 96-192 32-64-64 128 384', True, BLOCK_TYPE_INCEP),
            BlockConfig(1, '64 96-192 32-64-64 384', True, BLOCK_TYPE_INCEP),
            BlockConfig(1, '64 96-192 32-64-64 384', True, BLOCK_TYPE_INCEP),
            BlockConfig(1, '64 96-192 32-64-64 384', True, BLOCK_TYPE_INCEP)],
        scope = 'conv5',
        end_points = end_points)
    
    if include_last_bn_relu:
        with tf.variable_scope('conv5_4'):
            last_bn = slim.batch_norm(conv5, scope = 'last_bn')
            last_bn = __scale(last_bn, scope = 'last_bn')
            conv5 = tf.nn.relu(last_bn)
    end_points['conv5'] = conv5 
    return conv5, end_points

