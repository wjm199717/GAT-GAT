from __future__ import print_function
import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf
from utilities import run_random_walks_n2v
import dill

flags = tf.app.flags
FLAGS = flags.FLAGS
np.random.seed(123)


def load_graphs(dataset_str):
    """Load graph snapshots given the name of dataset"""
#np.load()专门用来加载.npy文件，.npy文件是一种用来保存数据的文件。
#graphs.npz文件里面存储的是一个graph文件（可以看作一个对象）。.npz文件是一个压缩文件，文件下面会有很多子文件
#graphs是一个列表，列表存储了图的静态快照，列表中的每个元素是一个图对象。
    graphs = np.load("data/{}/{}".format(dataset_str, "graphs.npz"), allow_pickle=True, encoding="latin1")['graph']
    print("Loaded {} graphs ".format(len(graphs)))
#map()会根据提供的函数对指定序列做映射。
#第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
#lambda后面接的是函数的形参
#python3的map函数返回的是一个迭代器，python2的map函数返回的是一个新列表
#adj_matrices可以看作是一个新的列表，列表中的每个元素为图对象转换后的邻接矩阵。    
    adj_matrices = list(map(lambda x: nx.adjacency_matrix(x), graphs))
    return graphs, adj_matrices


def load_feats(dataset_str):
    """ Load node attribute snapshots given the name of dataset (not used in experiments)"""
    features = np.load("data/{}/{}".format(dataset_str, "features.npz"), allow_pickle=True)['feats']
    print("Loaded {} X matrices ".format(len(features)))
    return features


def sparse_to_tuple(sparse_mx):
    """Convert scipy sparse matrix to tuple representation (for tf feed dict)."""
    #将邻接矩阵转化成元组表示[[],[],[],...]=[[(),a],[(),a],[(),a]...[(),a]]
    def to_tuple(mx):
#coo矩阵用来压缩系数矩阵，coo矩阵由三个数组组成，一个存放非零元素的值。
#剩下的两个数组，一个用来存放非零元素位置的行，一个用来存放非零元素位置的列。
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
#[row],[col](2*10)=[[第一非零元素行位置，第一非零元素列位置],[],...,[],[]](10*2)
        coords = np.vstack((mx.row, mx.col)).transpose()
        #values是一个列表[_,_,_,...,_]
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    def to_tuple_list(matrices):
        # Input is a list of matrices.
        coords = []
        values = []
        shape = [len(matrices)]
        for i in range(0, len(matrices)):
            mx = matrices[i]
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            # Create proper indices - coords is a numpy array of pairs of indices.
            #np.vstack((a,b))返回的是一个a和b堆叠的数组.
            coords_mx = np.vstack((mx.row, mx.col)).transpose()
            z = np.array([np.ones(coords_mx.shape[0]) * i]).T
#z就变为了[[_,_,_],[_,_,_],...,[_,_,_]]
            z = np.concatenate((z, coords_mx), axis=1)
            #astype(int)将数组z中的所有数据元素类型更换为int类型。
            z = z.astype(int)
#extend()函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
#coords = [[_,_,0],[_,_,1],[_,_,2],...,[_,_,n]]最后一个数字是几，就表示是第几个矩阵。
#values = [_,_,_,...,_]
            coords.extend(z)
            values.extend(mx.data)

#shape = [_,_,_] 三维
        shape.extend(matrices[0].shape)
        shape = np.array(shape).astype("int64")
        values = np.array(values).astype("float32")
        coords = np.array(coords)
        return coords, values, shape

#isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
    if isinstance(sparse_mx, list) and isinstance(sparse_mx[0], list):
        # Given a list of lists, convert it into a list of tuples.
        for i in range(0, len(sparse_mx)):
            sparse_mx[i] = to_tuple_list(sparse_mx[i])

    elif isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    #sum(1)将一个向量[_,_,...,_]全部元素加和。
    #rowsum = [_]
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_graph_gcn(adj):
    """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
#adj就变为了coo类型的存储矩阵
#sp.coo_matrix()用来创建稀疏矩阵，该函数返回的是一个矩阵。
    adj = sp.coo_matrix(adj)
#sp.eye函数返回一个adj.shape[0]*adj.shape[0]形状的对角矩阵，对角元素为1。
    adj_ = adj + sp.eye(adj.shape[0])
#先将adj_的元素进行加和，加和后的结果再加上1。    
    rowsum = np.array(adj_.sum(1))
#a.flatten()：a是个数组，a.flatten()就是把a降到一维，默认是按行的方向降。
#           ：a若是个矩阵，a.flatten()之后还是个矩阵[[_,_,...,_]]。
#power(x, y) 函数，计算 x 的 y 次方。
#    当 np.diag(array) 中
#    array是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵
#    array是一个二维矩阵时，结果输出矩阵的对角线元素
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    #dot()返回的是矩阵之间的乘积。（向量之间就是点积）（两个数之间就是乘法）
    #tocoo([copy])：返回稀疏矩阵的coo_matrix形式
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def get_context_pairs_incremental(graph):
    return run_random_walks_n2v(graph, graph.nodes())


def get_context_pairs(graphs, num_time_steps):
    """ Load/generate context pairs for each snapshot through random walk sampling."""
    load_path = "data/{}/train_pairs_n2v_{}.pkl".format(FLAGS.dataset, str(num_time_steps - 2))
    try:
#        dill 可以用于保存对象等大多数Python的数据格式
        context_pairs_train = dill.load(open(load_path, 'rb'))
        print("Loaded context pairs from pkl file directly")
    except (IOError, EOFError):
        print("Computing training pairs ...")
        context_pairs_train = []
        #num_time_steps表示训练的静态快照图，
        #所以这里的context_pairs_train列表包含num_time_steps个数的字典。
        for i in range(0, num_time_steps):
            #run_random_walks_n2v()返回的是一个字典。
            context_pairs_train.append(run_random_walks_n2v(graphs[i], graphs[i].nodes()))
        #保存不同时刻图的随机游走的固定窗口的节点对。
        dill.dump(context_pairs_train, open(load_path, 'wb'))
        print ("Saved pairs")

    return context_pairs_train


def get_evaluation_data(adjs, num_time_steps, dataset):
    """ Load train/val/test examples to evaluate link prediction performance"""
    eval_idx = num_time_steps - 2
    #.npz文件相当于一个文件夹，文件夹目录下面有许多文件。
    eval_path = "data/{}/eval_{}.npz".format(dataset, str(eval_idx))
    try:
        #空格加反斜杠是续行的意思，一行代码太长后面加上 \表示接下来的一行是跟着这一行的。
        #左斜杠/也叫正斜杠，右斜杠\又叫做反斜杠。
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
            np.load(eval_path, encoding='bytes', allow_pickle=True)['data']
        print("Loaded eval data")
    except IOError:
#adj是图的邻接矩阵，[[第一个图的邻接矩阵],[第二个图的邻接矩阵],[第三个图的邻接矩阵]]
        #next_adjs表示的是一个邻接矩阵
        next_adjs = adjs[eval_idx + 1]
        print("Generating and saving eval data ....")
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = create_data_splits(adjs[eval_idx], next_adjs, val_mask_fraction=0.2, test_mask_fraction=0.6)
#数据是以字典的形式存储在.npz文件中，这里键是data。     
        np.savez(eval_path, data=np.array([train_edges, train_edges_false, val_edges, val_edges_false,
                                           test_edges, test_edges_false]))

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def create_data_splits(adj, next_adj, val_mask_fraction=0.2, test_mask_fraction=0.6):
    """In: (adj, next_adj) along with test and val fractions. For link prediction (on all links), all links in
    next_adj are considered positive examples.
    Out: list of positive and negative pairs for link prediction (train/val/test)"""
#edges_all就是[[第一个非零元素行位置，第一个非零元素列位置],[],...,[],[]]
    edges_all = sparse_to_tuple(next_adj)[0]  # All edges in original adj.
#diagonal()查看矩阵对角线上的元素，返回矩阵对角线元素的列表。    
#np.newaxis的作用就是在这一位置增加一个一维。就是shape的形状多一个位置用1填充。
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)  # Remove diagonal elements
    #将adj矩阵的非零元素给消除掉，adj依旧是一个矩阵。
    adj.eliminate_zeros()
#adj.todense()返回一个矩阵。
#np.diag(a)将a数组元素变化为一个对角矩阵。
#当参数a是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵。[[],[],...,[]]
#当参数a是一个二维矩阵时，结果输出矩阵的对角线元素，[_,_,_,...,_]。
    #sum()该函数返回一个数。
    assert np.diag(adj.todense()).sum() == 0
    if next_adj is None:
        raise ValueError('Next adjacency matrix is None')

#list函数是替换的意思，将原先的数据类型去除掉，然后替换成list这种数据类型。
    edges_next = np.array(list(set(nx.from_scipy_sparse_matrix(next_adj).edges())))
    edges = []   # Constraint to restrict new links to existing nodes.
    #edges_next = [(_,_),(_,_),...,(_,_)]
    for e in edges_next:
        if e[0] < adj.shape[0] and e[1] < adj.shape[0]:
            edges.append(e)
    #edges = [(_,_),(_,_),...,(_,_)]
    edges = np.array(edges)

    def ismember(a, b, tol=5):
    #np.all是与操作，所有元素为true，则输出为true。
    #np.any是或操作，任意一个元素为true，则输出为true。
#round() 方法返回浮点数x的四舍五入值
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

#edges.shape[0]表示的是边的个数。
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    #用来测试的边数和用来验证的边数。
    num_test = int(np.floor(edges.shape[0] * test_mask_fraction))
    num_val = int(np.floor(edges.shape[0] * val_mask_fraction))
    #将图中的前num_val数量的边用来验证，后num_test数量的边用来测试。
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
#test_edges = [(),(),...,()]    val_edges = [(),(),...,()]    
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
#np.delete(_,_,_)表明删除目标向量中的子向量
#第一个参数为目标向量，第二个参数为删除子向量的位置，第三个参数表明沿着哪个维度删除。
#将一个图分为训练集，验证集，测试集。
#先将t3时刻的图的节点向量表示出来，然后用t3时刻更新的节点向量表示t4时刻节点的向量。
#也就是以t3时刻更新的节点向量来预测t4时刻哪些节点之间会有连边。
#将t4时刻的图分为训练集，验证集，测试集。以t3时刻更新的节点向量作为输入特征，用训练集训练一个分类器。
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # Create train edges.
    train_edges_false = []
    while len(train_edges_false) < len(train_edges):
        #adj.shape[0]表示的是adj这个图有多少个节点。
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            #continue退出整个循环。
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(train_edges_false)):
                continue
        #train_edges_false是[[_,_],[_,_],[_,_],...,[_,_]]
        train_edges_false.append([idx_i, idx_j])

    # Create test edges.
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    # Create val edges.
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue

        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)
    #无false的是正样本（正类），有false的是负样本（负类）。
    print("# train examples: ", len(train_edges), len(train_edges_false))
    print("# val examples:", len(val_edges), len(val_edges_false))
    print("# test examples:", len(test_edges), len(test_edges_false))

#为什么会有两个train_edges和train_edges_false,因为一个是正样本，一个是负样本。
    return list(train_edges), train_edges_false, list(val_edges), val_edges_false, list(test_edges), test_edges_false
