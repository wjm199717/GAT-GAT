#math_ops模块常用来做一些数字上的运算
from tensorflow.python.ops import math_ops
from inits import * 
import tensorflow as tf

#tf.layers.conv1d为一维卷积，一般用于处理文本数据，
#常用语自然语言处理中，输入一般是文本经过embedding的二维数据
conv1d = tf.layers.conv1d

#用来设置不同类型的命令行参数及其默认值，这个不同类型指的是字符串型，数值型，布尔性，浮点型
#FLAGS：定义一个全局对象来获取参数的值，在程序中使用(eg：FLAGS.iteration)来引用参数
flags = tf.app.flags
FLAGS = flags.FLAGS

#_LAYER_UIDS是一个字典
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """
#*args 用来将参数打包成tuple给函数体调用
#**kwargs 打包关键字参数成dict给函数体调用
    # *表示的是‘一系列’的意思
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        #kwargs代表的就是字典，这个字典是由**kwargs这个形参得到的
        for kwarg in kwargs.keys():
#assert用来判断表达式kwarg in allowed_kwargs是否为true，
#表达式如果为false，则报错，出现'Invalid keyword argument: ' + kwarg这段话
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        #get()函数得到字典中关键字为'name'所对应的值
        name = kwargs.get('name')
        #如果name为空字符串则执行if语句部分
        if not name:
#首先用self.__class__将实例变量指向类，然后再去调用__name__类属性
#__name__用来获取类名Layer，lower()将名字小写变成layer,是一个字符串
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        #初始化类的属性
        self.name = name
        self.vars = {}
#dict.get(key, default=None)
#key -- 字典中要查找的键。
#default -- 如果指定键的值不存在时，返回该默认值。
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False
#下划线前缀的含义是告知其他程序员：以单个下划线开头的变量或方法仅供内部使用
    def _call(self, inputs):
        return inputs
#这里的__call__就是一个方法名字，没有其他意义，加下划线是为了与上面的进行区别
    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
#tf.summary.histogram()在tensorflow可视化中用来显示直方图信息
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

#该类继承Layer类
class TemporalAttentionLayer(Layer):
    """ The input parameter num_time_steps is set as the total number of training snapshots +1."""
    def __init__(self, input_dim, n_heads, num_time_steps, attn_drop, residual=False, bias=True,
                 use_position_embedding=True, **kwargs):
        super(TemporalAttentionLayer, self).__init__(**kwargs)

        self.bias = bias
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.attn_drop = attn_drop
        self.attn_wts_means = []
        self.attn_wts_vars = []
        self.residual = residual
        self.input_dim = input_dim
#tf.contrib.layers是TensorFlow中的一个封装好的高级库
#xavier_initializer()该函数返回一个用于初始化权重的初始化器 “Xavier” 。
#Xavier这个初始化器是用来保持每一层的梯度大小都差不多相同。
        xavier_init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(self.name + '_vars'):
            if use_position_embedding:
                self.vars['position_embeddings'] = tf.get_variable('position_embeddings',
                                                                   dtype=tf.float32,
                                                                   shape=[self.num_time_steps, input_dim],
                                                                   initializer=xavier_init)  # [T, F]

            self.vars['Q_embedding_weights'] = tf.get_variable('Q_embedding_weights',
                                                               dtype=tf.float32,
                                                               shape=[input_dim, input_dim],
                                                               initializer=xavier_init)  # [F, F]
            self.vars['K_embedding_weights'] = tf.get_variable('K_embedding_weights',
                                                               dtype=tf.float32,
                                                               shape=[input_dim, input_dim],
                                                               initializer=xavier_init)  # [F, F]
            self.vars['V_embedding_weights'] = tf.get_variable('V_embedding_weights',
                                                               dtype=tf.float32,
                                                               shape=[input_dim, input_dim],
                                                               initializer=xavier_init)  # [F, F]

    def __call__(self, inputs):
        """ In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]."""
        # 1: Add position embeddings to input
#tf.tile是用来对张量进行扩张的，比如形状为[3,2]维的张量，进行[2,3]扩张，则扩张后的张量为[6,6]
#tf.expand_dims()该函数用来扩张输入张量的维度，以1来填充
        position_inputs = tf.tile(tf.expand_dims(tf.range(self.num_time_steps), 0), [tf.shape(inputs)[0], 1])
#tf.nn.embedding_lookup()的用法主要是选取一个张量里面索引对应的元素，索引[2,3]表示的是查找第2个和第3个元素
        temporal_inputs = inputs + tf.nn.embedding_lookup(self.vars['position_embeddings'],
                                                          position_inputs)  # [N, T, F]

        # 2: Query, Key based multi-head self attention.
#tf.tensordot(a, b, axes)
#tensordot函数用来进行矩阵相乘，它的一个好处是：当a和b的维度不同时，也可以相乘。
#tensordot([2,3,3],[3,2,6],axes=1),axes=1表示的是取第一个张量的最后一维，第二个张量的第一维
#tensordot([2,2,3],[3,2,6],axes=2),axes=2表示的是取第一个张量的后两维，第二个张量的前两维
        #这里的axes=[[2], [0]]表示的是temporal_inputs取第2维，self.vars['']取第0维
        #这里得到q,k,v矩阵,也就是索引矩阵，关键字矩阵，数值矩阵
        q = tf.tensordot(temporal_inputs, self.vars['Q_embedding_weights'], axes=[[2], [0]])  # [N, T, F]
        k = tf.tensordot(temporal_inputs, self.vars['K_embedding_weights'], axes=[[2], [0]])  # [N, T, F]
        v = tf.tensordot(temporal_inputs, self.vars['V_embedding_weights'], axes=[[2], [0]])  # [N, T, F]

        # 3: Split, concat and scale.
#tf.split()将一个张量进行划分，第一个参数为划分的张量，第二个参数为划分个数，第三个参数为沿哪个维度进行划分
        q_ = tf.concat(tf.split(q, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]
        k_ = tf.concat(tf.split(k, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]
        v_ = tf.concat(tf.split(v, self.n_heads, axis=2), axis=0)  # [hN, T, F/h]
        
        #tf.transpose函数用来交换维度，转置
        outputs = tf.matmul(q_, tf.transpose(k_, [0, 2, 1]))  # [hN, T, T]
        #一个节点通过时序注意力层后出来的特征数变为了self.num_time_steps
        outputs = outputs / (self.num_time_steps ** 0.5)

        # 4: Masked (causal) softmax to compute attention weights.
#tf.ones_like该操作返回一个具有和给定tensor相同形状（shape）和相同数据类型（dtype），
#但是所有的元素都被设置为1的tensor。
        diag_val = tf.ones_like(outputs[0, :, :])  # [T, T]
#tf.contrib.linalg用于线性计算的类，下面包括了好多线性操作的类，tf.linalg.LinearOperator()就是其中之一，
#还包括了如转置，奇异值分解，计算张量的迹等等关于张量线性运算的一些函数。
#LinearOperatorLowerTriangular()这个类是针对下三角张量的一些列操作
#下面这句话就是把diag_val转化成下三角张量
        #tf.contrib是一个模块，tril就是一个下三角张量
        tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [T, T]
#tensorflow中的tile()函数是用来对张量(Tensor)进行扩展的，其特点是对当前张量内的数据进行一定规则的复制。
#tf.expand_dims用来将张量的维度增加一维，比如shape为[2,3]的张量，维度增加一维变为[1,2,3],张量存储的数据量依然保持不变
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # [hN, T, T]
        padding = tf.ones_like(masks) * (-2 ** 32 + 1)
#tf.equal()判断张量中的元素与0元素是否相同，相同的位置就返回true，否则就返回false
#tf.where()将bool型张量中的true位置用padding中对应元素替换，bool型张量中的false位置用outputs中对应的元素替换        
        outputs = tf.where(tf.equal(masks, 0), padding, outputs)  # [h*N, T, T]
        outputs = tf.nn.softmax(outputs)  # Masked attention.
        self.attn_wts_all = outputs

        # 5: Dropout on attention weights.
#dropout：一种防止神经网络过拟合的手段。
#随机的拿掉网络中的部分神经元，从而减小对W权重的依赖，以达到减小过拟合的效果。
#注意：dropout只能用在训练中，测试的时候不能dropout，要用完整的网络测试哦。
        outputs = tf.layers.dropout(outputs, rate=self.attn_drop)
        outputs = tf.matmul(outputs, v_)  # [hN, T, C/h]

#因为是多头注意力层，所以得到的结果先需经过划分，然后将他们连接起来
        split_outputs = tf.split(outputs, self.n_heads, axis=0)
        outputs = tf.concat(split_outputs, axis=-1)

        # Optional: Feedforward and residual
        if FLAGS.position_ffn:
            outputs = self.feedforward(outputs)

        if self.residual:
            outputs += temporal_inputs

        return outputs

    def feedforward(self, inputs, reuse=None):
        """Point-wise feed forward net.

        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A 3d tensor with the same shape and dtype as inputs
        """
        with tf.variable_scope(self.name + '_vars', reuse=reuse):
            inputs = tf.reshape(inputs, [-1, self.num_time_steps, self.input_dim])
            params = {"inputs": inputs, "filters": self.input_dim, "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
#tf.layers.conv1d为一维卷积
#一维卷积一般用于处理文本数据，常用语自然语言处理中，输入一般是文本经过embedding的二维数据。
            outputs = tf.layers.conv1d(**params)
            outputs += inputs
        return outputs


class StructuralAttentionLayer(Layer):
    def __init__(self, input_dim, output_dim, n_heads, attn_drop, ffd_drop, act=tf.nn.elu, residual=False,
                 bias=True, sparse_inputs=False, **kwargs):
        #先初始化父类的属性，再初始化该类的属性
        super(StructuralAttentionLayer, self).__init__(**kwargs)
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.act = act
        self.bias = bias
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = residual
        self.sparse_inputs = sparse_inputs

        if self.logging:
            self._log_vars()
        self.n_calls = 0

    def _call(self, inputs):
        self.n_calls += 1
        x = inputs[0]
        adj = inputs[1]
        attentions = []
        reuse_scope = None
        for j in range(self.n_heads):
            if self.n_calls > 1:
                reuse_scope = True

            attentions.append(self.sp_attn_head(x, adj_mat=adj, in_sz=self.input_dim,
                                                out_sz=self.output_dim // self.n_heads, activation=self.act,
                                                in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=self.residual,
                                                layer_str="l_{}_h_{}".format(self.name, j),
                                                sparse_inputs=self.sparse_inputs,
                                                reuse_scope=reuse_scope))

        h = tf.concat(attentions, axis=-1)
        return h

    @staticmethod
    def leaky_relu(features, alpha=0.2):
        return math_ops.maximum(alpha * features, features)

    def sp_attn_head(self, seq, in_sz, out_sz, adj_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False,
                     layer_str="", sparse_inputs=False, reuse_scope=None):
        """ Sparse Attention Head for the GAT layer. Note: the variable scope is necessary to avoid
        variable duplication across snapshots"""
#reuse有三种取值，默认取值是None：
#True: 参数空间使用reuse 模式，即该空间下的所有tf.get_variable()函数将直接获取已经创建的变量，如果参数不存在tf.get_variable()函数将会报错。
#AUTO_REUSE：若参数空间的参数不存在就创建他们，如果已经存在就直接获取它们。
#None 或者False 这里创建函数tf.get_variable()函数只能创建新的变量，当同名变量已经存在时，函数就报错
        with tf.variable_scope('struct_attn', reuse=reuse_scope):
            if sparse_inputs:
                #reuse = False 所以tf.get_variable这里创建新的变量
                weight_var = tf.get_variable("layer_" + str(layer_str) + "_weight_transform", shape=[in_sz, out_sz],
                                             dtype=tf.float32)
                seq_fts = tf.expand_dims(tf.sparse_tensor_dense_matmul(seq, weight_var), axis=0)  # [N, F]
            else:
                seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False,
                                           name='layer_' + str(layer_str) + '_weight_transform', reuse=reuse_scope)

            # Additive self-attention.
            f_1 = tf.layers.conv1d(seq_fts, 1, 1, name='layer_' + str(layer_str) + '_a1', reuse=reuse_scope)
            f_2 = tf.layers.conv1d(seq_fts, 1, 1, name='layer_' + str(layer_str) + '_a2', reuse=reuse_scope)
            f_1 = tf.reshape(f_1, [-1, 1])  # [N, 1]
            f_2 = tf.reshape(f_2, [-1, 1])  # [N, 1]

            logits = tf.sparse_add(adj_mat * f_1, adj_mat * tf.transpose(f_2))  # adj_mat is [N, N] (sparse)
#tf.SparseTensor创建一个稀疏张量，indices这个参数是索引，指示非0元素所在地方
#value指的是具体数值，dense_shape是构建的张量形状，没有值的地方就用0来填充
            leaky_relu = tf.SparseTensor(indices=logits.indices,
                                         values=self.leaky_relu(logits.values),
                                         dense_shape=logits.dense_shape)
            coefficients = tf.sparse_softmax(leaky_relu)  # [N, N] (sparse)

            if coef_drop != 0.0:
                coefficients = tf.SparseTensor(indices=coefficients.indices,
                                               values=tf.nn.dropout(coefficients.values, 1.0 - coef_drop),
                                               dense_shape=coefficients.dense_shape)  # [N, N] (sparse)
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)  # [N, D]
#tf.squeeze()是删除张量中维度为1的维度。
            seq_fts = tf.squeeze(seq_fts)
            values = tf.sparse_tensor_dense_matmul(coefficients, seq_fts)
            values = tf.reshape(values, [-1, out_sz])
            #tf.expand_dims用来将维度增加一维，axis确定在哪里增加一维
            values = tf.expand_dims(values, axis=0)
            ret = values  # [1, N, F]

            if residual:
                residual_wt = tf.get_variable("layer_" + str(layer_str) + "_residual_weight", shape=[in_sz, out_sz],
                                              dtype=tf.float32)
                if sparse_inputs:
                    ret = ret + tf.expand_dims(tf.sparse_tensor_dense_matmul(seq, residual_wt),
                                               axis=0)  # [N, F] * [F, D] = [N, D].
                else:
                    ret = ret + tf.layers.conv1d(seq, out_sz, 1, use_bias=False,
                                                 name='layer_' + str(layer_str) + '_residual_weight', reuse=reuse_scope)
            return activation(ret)
