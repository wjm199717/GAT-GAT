from __future__ import division
from __future__ import print_function

import json
#python中的os模块包含普遍的操作系统功能。如果你希望你的程序能够与平台无关的话，这个模块尤其重要。
import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import networkx as nx
#logging模块实现日志功能
#logging模块是Python内置的标准模块，主要用于输出运行日志，可以设置输出日志的等级、日志保存路径、日志文件回滚等
import logging
#scipy函数库在numpy库的基础上增加了众多的数学，科学以及工程计算中常用的库函数。
import scipy

#用小数点.来进行文件的目录分级。
from link_prediction import evaluate_classifier, write_to_csv
from flags import *
from models import DySAT
import minibatch as um
import preprocess as up
import utilities as uu 


#np.random是一个随机数生成器，生成器（黑盒）的输入需要是一个种子，黑盒返回一个新种子和一个随机数。
#种子不同那么得到的随机数就不同，种子相同得到的随机数就相同，
#自己设置种子，这样的作用对后面的执行只有一次效果
#random是个复杂的随机数生成算法，也可以叫做随机数生成器。
#np.random.random()就是用np.random这个随机数生成器来生成随机数。
np.random.seed(123)
tf.set_random_seed(123)

#flags用来调用参数。
#用来设置不同类型的命令行参数及其默认值，这个不同类型指的是字符串型，数值型，布尔性，浮点型
#FLAGS：定义一个全局对象来获取参数的值，在程序中使用(eg：FLAGS.iteration)来引用参数
#在其他模块中用flags定义了变量，然后在一个不同的模块中调用已经定义好了的变量，需要重新申明flags,才能调用
flags = tf.app.flags
#FLAGS是一个对象，用来保存命令行参数的数据，
#下面这句话执行完毕以后，tensorflow自动将FLAGS对象保存的数据解析到FLAGS.__flags这个字典当中去。
FLAGS = flags.FLAGS
# Assumes a saved base model as input and model name to get the right directory.
output_dir = "./logs/{}_{}/".format(FLAGS.base_model, FLAGS.model)

#os.path.isdir用于判断对象是否为一目录
if not os.path.isdir(output_dir):
    #os.mkdir(path)为path对象创建一个目录
    os.mkdir(output_dir)

config_file = output_dir + "flags_{}.json".format(FLAGS.dataset)

#with open(config_file, 'r') as f:
#    config = json.load(f)
#    for name, value in config.items():
#FLAGS.__flags是一个字典，字典的键是flags参数里面的名字，值是参数名对应的值。
#        if name in FLAGS.__flags:
#            FLAGS.__flags[name].value = value
#FLAGS.flag_values_dict()将定义的参数解析成字典存储到FLAGS.__flags中
print("Updated flags", FLAGS.flag_values_dict().items())

# Set paths of sub-directories.
LOG_DIR = output_dir + FLAGS.log_dir
SAVE_DIR = output_dir + FLAGS.save_dir
CSV_DIR = output_dir + FLAGS.csv_dir
MODEL_DIR = output_dir + FLAGS.model_dir

if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

if not os.path.isdir(CSV_DIR):
    os.mkdir(CSV_DIR)

if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)

#os.environ是一个存储了当前环境信息的字典
#更改当前环境GPU的信息
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.GPU_ID)

#datetime.now()返回当前的时间
#strftime()返回时间对象的字符串形式
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
#datetime.today()也返回当前时间
today = datetime.today()

# Setup logging
log_file = LOG_DIR + '/%s_%s_%s_%s_%s.log' % (FLAGS.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day), str(FLAGS.time_steps))

#log_level的等级为info。
#以info这种级别来写入日志文件中。
log_level = logging.INFO
#basicConfig()是日志文件基本的设置，就是日志文件中出现的最基本的信息。
#利用logging.basicConfig()打印信息到控制台
#loggigng就是一个日志文件
#保存log到文件，如果在logging.basicConfig()设置filename 和filemode，则只会保存log到文件，不会输出到控制台。
logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
#flag_values_dict()是一个存储了参数信息的字典
#下面这句话就是向日志文件中写入超参数信息。
logging.info(FLAGS.flag_values_dict().items())

# Create file name for result log csv from certain flag parameters.
output_file = CSV_DIR + '/%s_%s_%s_%s.csv' % (FLAGS.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day))

# model_dir is not used in this code for saving.

# utils folder: utils.py, random_walk.py, minibatch.py
# models folder: layers.py, models.py
# main folder: train.py
# eval folder: link_prediction.py

"""
#1: Train logging format: Create a new log directory for each run (if log_dir is provided as input). 
Inside it,  a file named <>.log will be created for each time step. The default name of the directory is "log" and the 
contents of the <>.log will get appended per day => one log file per day.

#2: Model save format: The model is saved inside model_dir. 

#3: Output save format: Create a new output directory for each run (if save_dir name is provided) with embeddings at 
each 
time step. By default, a directory named "output" is created.

#4: Result logging format: A csv file will be created at csv_dir and the contents of the file will get over-written 
as per each day => new log file for each day.
"""

# Load graphs and features.

num_time_steps = FLAGS.time_steps

#graphs是一个列表，列表中的每一个元素是一个多重图对象，该列表一共有16个图元素，也就是有16个时刻的图。
#adjs是一个迭代器（也可以看作列表），列表中的每一个元素是一个图的邻接矩阵。
graphs, adjs = up.load_graphs(FLAGS.dataset)
if FLAGS.featureless:
#scipy.sparse是稀疏矩阵库
#shape[0]表示邻接矩阵的行数，也就是图上的节点数
#identity就是以节点数n创建一个n*n的单位矩阵，单位矩阵的每一行表示一个节点的one-hot向量。
#tocsr()将稀疏矩阵以一种压缩矩阵的方式进行存储。
#[range(0, x.shape[0]), :]将这个单位矩阵进行切片，这里就是复制一遍原来的单位矩阵
#scipy.sparse一般用于创建稀疏矩阵。
    #x.shape[0]代表的是图的节点数
    #adjs[num_time_steps - 1].shape[0]在这里代表的是图3的节点数
    #这里就是给图1创建一个单位矩阵，给图2创建一个单位矩阵，给图3创建一个单位矩阵
#feats=[[[图1节点1],[图1节点2],...,[图1节点n1]],[[图2节点1],[图2节点2],...,[图2节点n2]],[[图3节点1],[图3节点2],...,[图3节点n3]]]
    feats = [scipy.sparse.identity(adjs[num_time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in adjs if
             x.shape[0] <= adjs[num_time_steps - 1].shape[0]]
else:
    feats = up.load_feats(FLAGS.dataset)
#one-hot向量的维度为24
num_features = feats[0].shape[1]
#保证训练的图时间要小于最后一个时刻，这样进行下一时刻的预测才能有标签。
assert num_time_steps < len(adjs) + 1  # So that, (t+1) can be predicted.

adj_train = []
feats_train = []
num_features_nonzero = []
loaded_pairs = False

# Load training context pairs (or compute them if necessary)
#加载上下文节点是为了来优化神经网络的参数
#context_pairs_train是一个列表[{第一个图上下文节点对},{第二个图上下文节点对}]
context_pairs_train = up.get_context_pairs(graphs, num_time_steps)

# Load evaluation data.
train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
    up.get_evaluation_data(adjs, num_time_steps, FLAGS.dataset)

# Create the adj_train so that it includes nodes from (t+1) but only edges from t: this is for the purpose of
# inductive testing.
new_G = nx.MultiGraph()
#nodes函数返回的是一个有关节点的列表。
new_G.add_nodes_from(graphs[num_time_steps - 1].nodes(data=True))

#将graphs[2]这个图的全部节点放入到new_G这个自定义的新图中去。
#将graphs[1]这个图的全部边放入到new_G这个自定义的新图中去。
#edges()函数返回的是[[1,5],[1,6],...,[1,9]]
for e in graphs[num_time_steps - 2].edges():
    new_G.add_edge(e[0], e[1])

#graphs[2]
#adjs[2]
graphs[num_time_steps - 1] = new_G
adjs[num_time_steps - 1] = nx.adjacency_matrix(new_G)

print("# train: {}, # val: {}, # test: {}".format(len(train_edges), len(val_edges), len(test_edges)))
#将信息打印到控制台上。
logging.info("# train: {}, # val: {}, # test: {}".format(len(train_edges), len(val_edges), len(test_edges)))

# Normalize and convert adj. to sparse tuple format (to provide as input via SparseTensor)
#将adjs中的每一个邻接矩阵转换为元组。
adj_train = list(map(lambda adj: up.normalize_graph_gcn(adj), adjs))
#print(adj_train)


if FLAGS.featureless:  # Use 1-hot matrix in case of featureless.
#scipy.sparse.identity应该是创建一个单位矩阵，以csr的形式进行稀疏矩阵的存储。
#csr_matrix存储方式是用三个数组来进行稀疏矩阵的存储。
    feats = [scipy.sparse.identity(adjs[num_time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in feats if
             x.shape[0] <= feats[num_time_steps - 1].shape[0]]
#feats = [18*24的矩阵,23*24的矩阵,24*24的矩阵]
#num_features等于24
num_features = feats[0].shape[1]

#feats_train是一个新的列表，feats_train=[None,None,None]
#feats_train就是将稀疏矩阵以密集的方式进行存储，以此减少内存的消耗。
feats_train = list(map(lambda feat: up.preprocess_features(feat)[1], feats))
#print(feats_train)
#num_features_nonzero = [18, 23, 24]
#x[1].shape=一维数组的shape=(18,)             (23,)               (24,)
num_features_nonzero = [x[1].shape[0] for x in feats_train]
#print('非零特征数：',num_features_nonzero)

def construct_placeholders(num_time_steps):
    min_t = 0
    if FLAGS.window > 0:
        min_t = max(num_time_steps - FLAGS.window - 1, 0)
    placeholders = {
            #shape=(None,)就说明这个是个一维数组
        'node_1': [tf.placeholder(tf.int32, shape=(None,), name="node_1") for _ in range(min_t, num_time_steps)],
        # [None,1] for each time step.
        'node_2': [tf.placeholder(tf.int32, shape=(None,), name="node_2") for _ in range(min_t, num_time_steps)],
        # [None,1] for each time step.
        'batch_nodes': tf.placeholder(tf.int32, shape=(None,), name="batch_nodes"),  # [None,1]
        'features': [tf.sparse_placeholder(tf.float32, shape=(None, num_features), name="feats") for _ in
                     range(min_t, num_time_steps)],
        'adjs': [tf.sparse_placeholder(tf.float32, shape=(None, None), name="adjs") for i in
                 range(min_t, num_time_steps)],
        #shape=()表明这是一个数。
        'spatial_drop': tf.placeholder(dtype=tf.float32, shape=(), name='spatial_drop'),
        'temporal_drop': tf.placeholder(dtype=tf.float32, shape=(), name='temporal_drop')
    }
    return placeholders


print("Initializing session")
# Initialize session
#tf.ConfigProto()主要的作用是配置tf.Session的运算方式，比如gpu运算或者cpu运算
#config是一个类对象。
config = tf.ConfigProto()
#该句话的意思是使用GPU来进行运算。
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

#placeholders是一个字典
placeholders = construct_placeholders(num_time_steps)

#这里minibatchIterator只是一个对象。
minibatchIterator = um.NodeMinibatchIterator(graphs, feats_train, adj_train,
                                          placeholders, num_time_steps, batch_size=FLAGS.batch_size,
                                          context_pairs=context_pairs_train)
#minibatchIterator.num_training_batches()返回一次迭代需要进行训练的批次数。
print("# training batches per epoch", minibatchIterator.num_training_batches())

model = DySAT(placeholders, num_features, num_features_nonzero, minibatchIterator.degs)
sess.run(tf.global_variables_initializer())

# Result accumulator(累加器) variables.
#epochs_rest_result和epochs_val_result都是字典。
epochs_test_result = uu.defaultdict(lambda: [])
epochs_val_result = uu.defaultdict(lambda: [])
epochs_embeds = []
epochs_attn_wts_all = []

for epoch in range(FLAGS.epochs):
    minibatchIterator.shuffle()
    epoch_loss = 0.0
    it = 0
    print('Epoch: %04d' % (epoch + 1))
    epoch_time = 0.0
    #end()用来判断当前批次是否到达了训练数据的尺寸。
    while not minibatchIterator.end():
        # Construct feed dictionary
        #feed_dict是一个字典。
        feed_dict = minibatchIterator.next_minibatch_feed_dict()
        feed_dict.update({placeholders['spatial_drop']: FLAGS.spatial_drop})
        feed_dict.update({placeholders['temporal_drop']: FLAGS.temporal_drop})
        t = time.time()
        # Training step
        #model是一个DySAT类对象。
        _, train_cost, graph_cost, reg_cost = sess.run([model.opt_op, model.loss, model.graph_loss, model.reg_loss],
                                                       feed_dict=feed_dict)
        #time.time()返回当前时间的时间戳。
        epoch_time += time.time() - t
        # Print results
        #将内容写在日志文件中。
        logging.info("Mini batch Iter: {} train_loss= {:.5f}".format(it, train_cost))
        logging.info("Mini batch Iter: {} graph_loss= {:.5f}".format(it, graph_cost))
        logging.info("Mini batch Iter: {} reg_loss= {:.5f}".format(it, reg_cost))
        logging.info("Time for Mini batch : {}".format(time.time() - t))

        epoch_loss += train_cost
        it += 1

    print("Time for epoch ", epoch_time)
    logging.info("Time for epoch : {}".format(epoch_time))
    if (epoch + 1) % FLAGS.test_freq == 0:
        minibatchIterator.test_reset()
        emb = []
        feed_dict.update({placeholders['spatial_drop']: 0.0})
        feed_dict.update({placeholders['temporal_drop']: 0.0})
        if FLAGS.window < 0:
            #final_output_embeddings是一个属性
            assert FLAGS.time_steps == model.final_output_embeddings.get_shape()[1]
        emb = sess.run(model.final_output_embeddings, feed_dict=feed_dict)[:,
              model.final_output_embeddings.get_shape()[1] - 2, :]
        emb = np.array(emb)
        # Use external classifier to get validation and test results.
        val_results, test_results, _, _ = evaluate_classifier(train_edges,
                                                              train_edges_false, val_edges, val_edges_false, test_edges,
                                                              test_edges_false, emb, emb)

        epoch_auc_val = val_results["HAD"][1]
        epoch_auc_test = test_results["HAD"][1]

        print("Epoch {}, Val AUC {}".format(epoch, epoch_auc_val))
        print("Epoch {}, Test AUC {}".format(epoch, epoch_auc_test))
        logging.info("Val results at epoch {}: Measure ({}) AUC: {}".format(epoch, "HAD", epoch_auc_val))
        logging.info("Test results at epoch {}: Measure ({}) AUC: {}".format(epoch, "HAD", epoch_auc_test))

        epochs_test_result["HAD"].append(epoch_auc_test)
        epochs_val_result["HAD"].append(epoch_auc_val)
        epochs_embeds.append(emb)
    epoch_loss /= it
    print("Mean Loss at epoch {} : {}".format(epoch, epoch_loss))

# Choose best model by validation set performance.
best_epoch = epochs_val_result["HAD"].index(max(epochs_val_result["HAD"]))

print("Best epoch ", best_epoch)
logging.info("Best epoch {}".format(best_epoch))

val_results, test_results, _, _ = evaluate_classifier(train_edges, train_edges_false, val_edges, val_edges_false,
                                                      test_edges, test_edges_false, epochs_embeds[best_epoch],
                                                      epochs_embeds[best_epoch])

print("Best epoch val results {}\n".format(val_results))
print("Best epoch test results {}\n".format(test_results))

logging.info("Best epoch val results {}\n".format(val_results))
logging.info("Best epoch test results {}\n".format(test_results))

write_to_csv(val_results, output_file, FLAGS.model, FLAGS.dataset, num_time_steps, mod='val')
write_to_csv(test_results, output_file, FLAGS.model, FLAGS.dataset, num_time_steps, mod='test')

# Save final embeddings in the save directory.
emb = epochs_embeds[best_epoch]
np.savez(SAVE_DIR + '/{}_embs_{}_{}.npz'.format(FLAGS.model, FLAGS.dataset, FLAGS.time_steps - 2), data=emb)
