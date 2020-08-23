from __future__ import print_function
import numpy as np
import networkx as nx
import tensorflow as tf
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from random_walk import Graph_RandomWalk

flags = tf.app.flags
FLAGS = flags.FLAGS


def to_one_hot(labels, N, multilabel=False):
    """In: list of (nodeId, label) tuples, #nodes N
       Out: N * |label| matrix"""
    ids, labels = zip(*labels)
    lb = MultiLabelBinarizer()
    if not multilabel:
        labels = [[x] for x in labels]
    lbs = lb.fit_transform(labels)
    encoded = np.zeros((N, lbs.shape[1]))
    for i in range(len(ids)):
        encoded[ids[i]] = lbs[i]
    return encoded


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


"""Random walk-based pair generation."""

def run_random_walks_n2v(graph, nodes, num_walks=10, walk_len=40):
    """ In: Graph and list of nodes
        Out: (target, context) pairs from random walk sampling using the sampling strategy of node2vec (deepwalk)"""
    #walk_len为40
    walk_len = FLAGS.walk_len
    #nx_G为一个创建的新图对象
    nx_G = nx.Graph()
    #将图对象转换成图的邻接矩阵
    #这里的邻接矩阵是带有权重的邻接矩阵。
    adj = nx.adjacency_matrix(graph)
    #多重图对象有这个edges()方法
    #这里的一个e是一个元组
    for e in graph.edges():
        #e[0],e[1]代表的是节点。
        #通过添加输入图graph的边到新图上面。
        nx_G.add_edge(e[0], e[1])

    for edge in graph.edges():
        #nx_G[edge[0]][edge[1]]出来的是两个节点连边的信息
        #nx_G[edge[0]]出来的是有关这个节点连边的信息，weight是连边的属性名。
        #adj[edge[0], edge[1]]的含义是两个节点间连边的权重。
        #nx_G[edge[0]][edge[1]]返回的是两个节点连边的信息的字典。
        #nx_G该图与传入的graph图是同一个图。
        nx_G[edge[0]][edge[1]]['weight'] = adj[edge[0], edge[1]]

#无向图上进行随机游走采样
#Graph_RandomWalk是一个类，G是该类的类对象。
    G = Graph_RandomWalk(nx_G, False, 1.0, 1.0)
    G.preprocess_transition_probs()
    #walks是一个列表，[[first walk],[second walk],[third walk]]
    #随机游走采用的方式是别名采样方法
    walks = G.simulate_walks(num_walks, walk_len)
    #这个WINDOW_SIZE = 10是以目标节点为中心，左边的节点数小于等于10，
    #右边的节点数小于等于10。
    WINDOW_SIZE = 10
#defaultdict()作用：以defaultdict定义的字典中，若查找某个键不存在时，则返回lambda: []，也就是返回一个空列表。
#defaultdict()当查找的键不存在于字典中时，程序不会报错，而是返回其中的参数（工厂函数）
    #这里的pairs就是一个字典
    pairs = defaultdict(lambda: [])
    pairs_cnt = 0
    for walk in walks:
#每个walk里面的每个节点，都找到该节点的上下文节点，把这些节点存储在一个字典中。
        #enumerate返回的是一个迭代器对象，这个对象包含了列表的元素和对应的索引。
        for word_index, word in enumerate(walk):
            for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(walk)) + 1]:
                if nb_word != word:
                    pairs[word].append(nb_word)
                    pairs_cnt += 1
    print("# nodes with random walk samples: {}".format(len(pairs)))
    print("# sampled pairs: {}".format(pairs_cnt))
#pairs这个字典包含了全部随机游走的全部目标节点的上下文节点，{目标节点：上下文节点}
    return pairs
