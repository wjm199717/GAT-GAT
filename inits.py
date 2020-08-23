import tensorflow as tf
import numpy as np
#下面的代码时有关各种初始化的
# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/tkipf/gcn

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
#均匀分布初始化
#tf.random_uniform()返回一个shape形状的张量，张量里元素取值位于[-scale,scale]
#张量里元素取值是均匀分布取值
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
#这个函数返回的是一个变量，而这个变量就是开始随机初始化的张量   
    return tf.Variable(initial, name=name)

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    #np.sqrt()对参数进行开方
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
