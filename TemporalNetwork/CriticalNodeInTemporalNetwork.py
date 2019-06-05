#coding=utf-8

import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf


def FRINetworks():
    G = nx.DiGraph().to_undirected()
    begin = datetime.datetime.fromtimestamp(1353303380)
    edges = dict((k,[]) for k in range(30))
    for line in open('highschool_2012.csv', 'r').readlines():
        contents = line.strip().split('\t')[:3]
        G.add_nodes_from([contents[1], contents[2]])
        edges[int((datetime.datetime.fromtimestamp(float(contents[0]))-begin).total_seconds()/24361)].append((contents[1], contents[2]))
    FRI = []
    for i in range(30):
        g = G.copy()
        g.add_edges_from(edges[i])
        FRI.append(g)
    return FRI

def ContactNetworks():
    G = nx.DiGraph().to_undirected()
    begin = datetime.datetime.fromtimestamp(21574)
    edges = dict((k,[]) for k in range(30))
    for line in open('out.contact', 'r').readlines():
        contents = line.strip().replace(' ', '\t').split('\t')[:4]
        G.add_nodes_from([contents[0], contents[1]])
        edges[int((datetime.datetime.fromtimestamp(float(contents[3]))-begin).total_seconds()/11500)].append((contents[0], contents[1]))
    FRI = []
    for i in range(30):
        g = G.copy()
        g.add_edges_from(edges[i])
        FRI.append(g)
    return FRI

def HypertextNetworks():
    G = nx.DiGraph().to_undirected()
    begin = datetime.datetime.fromtimestamp(1246255220)
    edges = dict((k,[]) for k in range(30))
    for line in open('out.hypertext', 'r').readlines():
        contents = line.strip().replace(' ', '\t').split('\t')[:4]
        G.add_nodes_from([contents[0], contents[1]])
        edges[int((datetime.datetime.fromtimestamp(float(contents[3]))-begin).total_seconds()/7080)].append((contents[0], contents[1]))
    FRI = []
    for i in range(30):
        g = G.copy()
        g.add_edges_from(edges[i])
        FRI.append(g)
    return FRI

def InfectiousNetworks():
    G = nx.DiGraph().to_undirected()
    begin = datetime.datetime.fromtimestamp(1247652139)
    edges = dict((k,[]) for k in range(30))
    for line in open('out.infectious', 'r').readlines():
        contents = line.strip().replace(' ', '\t').split('\t')[:4]
        G.add_nodes_from([contents[0], contents[1]])
        edges[int((datetime.datetime.fromtimestamp(float(contents[3]))-begin).total_seconds()/950)].append((contents[0], contents[1]))
    FRI = []
    for i in range(30):
        g = G.copy()
        g.add_edges_from(edges[i])
        FRI.append(g)
    return FRI

def DNCNetworks():
    G = nx.DiGraph().to_undirected()
    begin = datetime.datetime.fromtimestamp(1421071197)
    edges = dict((k,[]) for k in range(30))
    for line in open('out.dnc', 'r').readlines():
        contents = line.strip().replace(' ', '\t').split('\t')[:4]
        G.add_nodes_from([contents[0], contents[1]])
        edges[int((datetime.datetime.fromtimestamp(float(contents[3]))-begin).total_seconds()/1440000)].append((contents[0], contents[1]))
    FRI = []
    for i in range(30):
        g = G.copy()
        g.add_edges_from(edges[i])
        FRI.append(g)
    return FRI

def kendall(array1,array2):
    sum = 0
    n=len(array1)
    for i in range(len(array1)):
        p=0
        for j in range(i+1,len(array1)):
            if array2[i]>array2[j]:
                p=p+1
        sum=sum+p
    x = float(4*sum)/(n*(n-1))-1
    return x

def TemporalDegree(temporalG):#时序度
    N = len(temporalG[0].nodes())
    L = len(temporalG)
    degree = dict((k, 0) for k in temporalG[0].nodes())
    for t in range(L):
        for node in degree.keys():
            degree[node] += temporalG[t].degree(node)
    return sorted(degree.items(), key=lambda x: x[1], reverse=True)

def TemporalDegreeDeviation(temporalG):#时序度
    N = len(temporalG[0].nodes())
    L = len(temporalG)
    degree = dict((k, []) for k in temporalG[0].nodes())
    TDD = dict((k, []) for k in temporalG[0].nodes())
    for t in range(L):
        for node in degree.keys():
            degree[node].append(temporalG[t].degree(node))
    for node in degree.keys():
        tempTDD = 0
        miu = np.mean(degree[node])
        for d in degree[node]:
            tempTDD += (d - miu)**2
        TDD[node] = (tempTDD/L)**0.5
    return sorted(TDD.items(), key=lambda x: x[1], reverse=True)


def TemporalKshell(temporalG):#时序Kshell
    W = {}
    tk = dict((k, 0) for k in temporalG[0].nodes())
    for G in temporalG:
        tempKshell = nx.core_number(G)
        for edge in G.edges():
            w = min(tempKshell[edge[0]], tempKshell[edge[1]])
            if edge in W.keys():
                W[edge] += w
            else:
                W[edge] = w
    for edge in W.keys():
        tk[edge[0]] += W[edge]
        tk[edge[1]] += W[edge]
    return sorted(tk.items(), key=lambda x: x[1], reverse=True)

def TemporalCloseness(temporalG):
    L = len(temporalG)
    Closeness = dict((k, 0) for k in temporalG[0].nodes())
    #初始化pair对，每个快照有一个pair对
    nodePairs = {}
    for u in Closeness.keys():
        for v in Closeness.keys():
            if u!=v:
                nodePairs[(u, v)] = float('inf')
            else:
                nodePairs[(u, v)] = 0
    detaT = []
    for t in range(L):
        detaT.append(nodePairs.copy())
    #最后一个快照的pair对距离
    allDistance = dict(nx.all_pairs_shortest_path_length(temporalG[L-1]))
    for u in temporalG[0].nodes():
        for v in allDistance[u].keys():
            if allDistance[u][v] !=0:
                Closeness[u] += 1.0/allDistance[u][v]
    for pair in detaT[L-1].keys():
        if pair[1] in allDistance[pair[0]].keys():
            detaT[L-1][pair] = allDistance[pair[0]][pair[1]]
        else:
            detaT[L - 1][pair] = float('inf')

    for t in range(L-1)[::-1]:
        for pair in detaT[t].keys():
            if pair[0] != pair[1]:
                for neighbor in list(temporalG[t].neighbors(pair[0]))+[pair[0]]:
                    tempD = detaT[t+1][(neighbor, pair[1])] + 1
                    if tempD < detaT[t][pair]:
                        detaT[t][pair] = tempD
                Closeness[pair[0]] += 1.0/detaT[t][pair]

    return sorted(Closeness.items(), key=lambda x: x[1], reverse=True)

def TemporalBetweenness(temporalG):
    L = len(temporalG)
    Betweenness = dict((k, 0) for k in temporalG[0].nodes())
    #初始化pair对，每个快照有一个pair对
    nodePairs = {}
    for u in Betweenness.keys():
        for v in Betweenness.keys():
            if u == v:
                nodePairs[(u, v)] = [[u]]
            else:
                nodePairs[(u, v)] = []
    detaT = []
    for t in range(L):
        detaT.append(nodePairs.copy())
    #最后一个快照的pair对距离
    N = 0
    for pair in detaT[L-1].keys():
        if pair[0] != pair[1]:
            try:
                detaT[L-1][pair] = [p for p in nx.all_shortest_paths(temporalG[L-1], pair[0], pair[1])]
            except nx.exception.NetworkXNoPath:
                detaT[L - 1][pair] = []
            for path in detaT[L-1][pair]:
                if len(path)>2:
                    for node in path[1:-1]:
                        Betweenness[node] += 1.0/len(detaT[L-1][pair])
    for t in range(L-1)[::-1]:
        for pair in detaT[t].keys():
            if pair[0] != pair[1]:
                for neighbor in list(temporalG[t].neighbors(pair[0]))+[pair[0]]:
                    if detaT[t+1][(neighbor, pair[1])]!=[]:
                        tempL = len(detaT[t+1][(neighbor, pair[1])][0])
                        if detaT[t][pair] != []:
                            if tempL < len(detaT[t][pair][0]) - 1:
                                detaT[t][pair] = []
                                for path in detaT[t + 1][(neighbor, pair[1])]:
                                    detaT[t][pair].append([pair[0]] + path)
                            elif tempL == len(detaT[t][pair][0]) - 1:
                                for path in detaT[t + 1][(neighbor, pair[1])]:
                                    detaT[t][pair].append([pair[0]] + path)
                        else:
                            for path in detaT[t+1][(neighbor, pair[1])]:
                                detaT[t][pair].append([pair[0]]+path)
                for path in detaT[t][pair]:
                    if len(path) > 2:
                        for node in path[1:-1]:
                            if node != path[0] and node != path[-1]:
                                Betweenness[node] += 1.0 /len(detaT[t][pair])

    return sorted(Betweenness.items(), key=lambda x: x[1], reverse=True)

def TemporalDC(temporalG, beta, miu):
    L = len(temporalG)
    nodeSocre = dict((k, 0) for k in temporalG[0].nodes())
    N = len(nodeSocre)
    S = 0
    V = np.ones((N,1))
    for t in range(L):
        H = 1
        for alpha in range(t)[::-1]:
            tempA = nx.adjacency_matrix(temporalG[alpha], nodelist=[x for x in temporalG[0].nodes()]).todense()
            H = H*(beta*tempA+(1-miu)*np.eye(N))
        A= nx.adjacency_matrix(temporalG[t], nodelist=[x for x in temporalG[0].nodes()]).todense()
        S += beta*H*A*V
    for i in range(len(temporalG[0].nodes())):
        nodeSocre[list(temporalG[0].nodes())[i]] = S[i, 0]

    return sorted(nodeSocre.items(), key=lambda x: x[1], reverse=True)

def InfluenceModel(temporalG, beta, miu):
    L = len(temporalG)
    Influence = dict((k, 1) for k in temporalG[0].nodes())
    detaT = []
    for i in range(L):
        detaT.append(Influence.copy())
    for node in detaT[L-1].keys():
        detaT[L - 1][node] = beta*temporalG[L-1].degree(node)
    for t in range(L-1)[::-1]:
        for node in detaT[t].keys():
            detaT[t][node] += (1-miu)*detaT[t+1][node]
            for neighbor in temporalG[t].neighbors(node):
                detaT[t][node] += beta*detaT[t+1][neighbor]
    return sorted(detaT[0].items(), key=lambda x: x[1], reverse=True)

def SIR(temporalG, infected, beta, miu):
    N = 1000
    re = 0
    while N > 0:
        inf = set()
        inf.add(infected)
        R = set()
        for G in temporalG:
            newInf = []
            for i in inf:
                for j in G.neighbors(i):
                    k = random.uniform(0,1)
                    if k < beta and j not in inf and j not in R:
                        newInf.append(j)
                k2 = random.uniform(0, 1)
                if k2 >miu:
                    newInf.append(i)
                else:
                    R.add(i)
            inf = set(newInf)
        re += len(R)+len(inf)
        N -= 1
    return re/1000.0

def unionNetworks(graphs):
    if len(graphs) == 1:
        return graphs[0]
    G = nx.DiGraph().to_undirected()
    G.add_nodes_from(graphs[0].nodes())
    for g in graphs[1:]:
        G.add_edges_from(g.edges())
    return G

def accumulateNetworks(temporalG, a):
    newG = []
    N = len(temporalG)
    for i in range(N/a):
        newG.append(unionNetworks(temporalG[i*a:(i+1)*a]))
    if len(temporalG[(i+1)*a:]) > 0:
        newG.append(unionNetworks(temporalG[(i+1)*a:]))
    return newG

def chooseNetworks(temporalG, a):
    newG = []
    N = len(temporalG)
    for i in range(N/a):
        newG.append(temporalG[i*a])
    return newG

def ALLSIR(temporalG, beta, miu):
    re = {}
    for node in temporalG[0].nodes():
        re[node] = SIR(temporalG, node, beta, miu)
    out = open('SIR_TBA_'+str(beta)+'_'+str(miu), 'w')
    for i in re.keys():
        out.writelines(i+' '+str(re[i])+'\n')
    out.close()
    return re

def readSIR(fileName):
    re = {}
    for line in open(fileName).readlines():
        content = line.strip().split(' ')
        re[content[0]] = float(content[1])
    return re

def drawA(temporalG, beta, miu):
    x = range(1, 11)
    TD = []
    TC = []
    TB = []
    TK = []

    allSIR = ALLSIR(temporalG, beta, miu)
    for i in x:
        G = accumulateNetworks(temporalG, i)

        re = []
        for node in TemporalDegree(G):
            re.append(allSIR[node[0]])
        TD.append(kendall(range(len(re)), re))

        re = []
        for node in TemporalCloseness(G):
            re.append(allSIR[node[0]])
        TC.append(kendall(range(len(re)), re))

        re = []
        for node in TemporalBetweenness(G):
            re.append(allSIR[node[0]])
        TB.append(kendall(range(len(re)), re))

        re = []
        for node in TemporalKshell(G, beta, miu):
            re.append(allSIR[node[0]])
        TK.append(kendall(range(len(re)), re))

    print('TD',TD)
    print('TC', TC)
    print('TB', TB)
    print('TK', TK)

    plt.plot(x, TD, 'r', label='TD', marker='o', markersize=10)
    plt.plot(x, TC, 'b', label='TC', marker='s', markersize=10)
    plt.plot(x, TB, 'y', label='TB', marker='*', markersize=10)
    plt.plot(x, TK, 'k', label='TK', marker='^', markersize=10)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(ymax=1, ymin=0)
    plt.xlabel('L', fontsize=20)
    plt.ylabel('c', fontsize=20)
    plt.legend(fontsize=18)
    plt.show()

def drawB(temporalG, beta, miu):
    x = range(1, 11)
    TD = []
    TC = []
    TB = []
    TDCM = []
    allSIR = ALLSIR(temporalG, beta, miu)
    for i in x:
        G = chooseNetworks(temporalG, i)

        re = []
        for node in TemporalDegree(G):
            re.append(allSIR[node[0]])
        TD.append(kendall(range(len(re)), re))

        re = []
        for node in TemporalCloseness(G):
            re.append(allSIR[node[0]])
        TC.append(kendall(range(len(re)), re))

        re = []
        for node in TemporalBetweenness(G):
            re.append(allSIR[node[0]])
        TB.append(kendall(range(len(re)), re))

        re = []
        for node in TDC(G, beta, miu):
            re.append(allSIR[node[0]])
        TDCM.append(kendall(range(len(re)), re))

    print('TD',TD)
    print('TC', TC)
    print('TB', TB)
    print('TDCM', TDCM)

    plt.plot(x, TD, 'r', label='TD', marker='o', markersize=10)
    plt.plot(x, TC, 'b', label='TC', marker='s', markersize=10)
    plt.plot(x, TB, 'y', label='TB', marker='*', markersize=10)
    plt.plot(x, TDCM, 'k', label='TDCM', marker='^', markersize=10)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(ymax=1, ymin=0)
    plt.xlabel('L', fontsize=20)
    plt.ylabel('c', fontsize=20)
    plt.legend(fontsize=18)
    plt.show()

def NE(X, T, N, A, D):
    #计算邻接矩阵A，度矩阵D，
    tempX = np.zeros((N, T))
    temp = np.matrix(np.ones((N, 1)))
    for col in range(T-1):
        tempA = np.matrix(A[col])
        tempD = np.matrix(D[col])
        # tempM = tempD*tempA*tempD*np.matrix(X)[:, col]
        tempM = tempD * tempA * tempD * temp
        for row in range(N):
            tempX [row, col] = tempM[row]
    return tempX

def calAD(temporalG, N):
    A = []
    D = []
    for G in temporalG:
        tempA = np.zeros((N, N))
        tempD = np.zeros((N, N))
        for i in range(N):
            Di = 0
            for j in range(N):
                if G.has_edge(str(i), str(j)):
                    tempA[i, j] = 1
                    Di += 1
                elif i == j:
                    tempA[i, j] = 1
            if Di == 0:
                tempD[i, i] = 0
            else:
                tempD[i, i] = Di ** 0.5
        A.append(tempA)
        D.append(tempD)
    return A,D



def MLtest(data_train, data_test, N):
    X_train = data_train[:, 1:-1]
    Y_train = data_train[:, -1]
    X_test = data_test[:, 1:-1]
    data_test_sorted = data_test[data_test[:, -1].argsort()][::-1,:]

    from sklearn import linear_model

    clf = linear_model.LinearRegression()
    clf.fit(X_train, Y_train)

    return len(set(data_test[clf.predict(X_test).argsort()][::-1,:][:int(N*0.05),0]) & set(data_test_sorted[:int(N*0.05), 0]))/float(int(N*0.05)),\
len(set(data_test[clf.predict(X_test).argsort()][::-1,:][:int(N*0.1),0]) & set(data_test_sorted[:int(N*0.1), 0]))/float(int(N*0.1))

def DP(data_train, data_test, N, NumberOfFeature):
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    for x in data_train:
        train_X.append(x[1:NumberOfFeature+1])
        train_y.append([x[NumberOfFeature+1]])
    for x in data_test:
        test_X.append(x[1:NumberOfFeature+1])
        test_y.append([x[NumberOfFeature+1]])
    train_X = np.array(train_X, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.float32)
    test_X = np.array(test_X, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.float32)

    # 定义神经网络参数
    w1 = tf.Variable(tf.random_normal((NumberOfFeature, 32), stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal((32, 1), stddev=1, seed=1))

    x = tf.placeholder(tf.float32, shape=(None, NumberOfFeature), name='x-input')
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

    # 前向传播
    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    # 损失函数和反向传播
    mse = tf.reduce_mean(tf.square(y_ - y))
    train_step = tf.train.AdamOptimizer(0.001).minimize(mse)

    # 创建一个会话来运行tensorflow程序
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        # 初始化变量
        sess.run(init_op)


        # 设定训练的轮数
        STEPS = 3000
        for i in range(STEPS):
            # 每次选区batchsize的样本进行训练
            sess.run(train_step, feed_dict={x: train_X, y_: train_y})

            if i % 1000 == 0:
                # 每隔一段时间计算交叉熵并输出
                total_mse = sess.run(mse, feed_dict={x: train_X, y_: train_y})
                # print('After %d training step(s), mse on all data is %g' % (i, total_mse))

        predict = sess.run(y, feed_dict={x: test_X, y_: test_y})
        re = []
        real = []
        for i in range(len(data_test)):
            re.append((data_test[i][0], predict[i]))
            real.append((data_test[i][0], data_test[i][-1]))
        re = [x[0] for x in sorted(re, key=lambda x: x[1], reverse=True)]
        real = [x[0] for x in sorted(real, key=lambda x: x[1], reverse=True)]

        return len(set(re[:int(N*0.05)]) & set(real[:int(N*0.05)])) / float(N*0.05), \
               len(set(re[:int(N * 0.1)]) & set(real[:int(N * 0.1)])) / float(N * 0.1)


def makeDataDegree(G, beta, miu, name):
    Fect = {}
    for i in range(30):
        BA = G[i]
        if i == 0:
            for node in BA.nodes():
                Fect[node] = [node]
        for node in BA.nodes():
            Fect[node].append(float(BA.degree(node)))
    for node in G[0].nodes():
        Fect[node].append(SIR(G, node, beta, miu))

    X = np.array([x for x in Fect.values()], dtype=float)
    np.save("data_"+name+"_D_"+str(beta)+"_"+str(miu)+".npy", X)

    return True

def makeDataAll(G, beta, miu, name):
    Fect = {}
    for i in range(30):
        BA = G[i]
        if i == 0:
            for node in BA.nodes():
                Fect[node] = [node]

        Kshell = nx.core_number(BA)

        # allNeighborsDegree = {}
        # for node in BA.nodes():
        #     allNeighborsDegree[node] = 0
        #     for neighbor in BA.neighbors(node):
        #         allNeighborsDegree[node] += BA.degree(neighbor)


        for node in BA.nodes():
            #degree
            Fect[node].append(float(BA.degree(node)))
            #kshell
            Fect[node].append(float(Kshell[node]))
    for node in G[0].nodes():
        Fect[node].append(SIR(G, node, beta, miu))

    X = np.array([x for x in Fect.values()], dtype=float)
    np.save("NewData/data_"+name+"_D_"+str(beta)+"_"+str(miu)+".npy", X)

    return True


def Top(data_train,data_test, N, allT):
    t1 = int(N*0.05)
    t2 = int(N * 0.1)
    X = data_test[data_test[:, -1].argsort()][::-1, :][:, 0]
    TC, TB, TK, TDD, TDC = allT
    NN = DP(data_train, data_test, N, 30)
    LM = MLtest(data_train, data_test, N)

    top5=([LM[0], NN[0],len(set(X[:t1]) & set(TC[:t1])) / float(t1),
          len(set(X[:t1]) & set(TB[:t1])) / float(t1), len(set(X[:t1]) & set(TK[:t1])) / float(t1)
              , len(set(X[:t1]) & set(TDD[:t1])) / float(t1)
              , len(set(X[:t1]) & set(TDC[:t1])) / float(t1)])

    top10 = ([LM[1], NN[1], len(set(X[:t2]) & set(TC[:t2])) / float(t2),
            len(set(X[:t2]) & set(TB[:t2])) / float(t2), len(set(X[:t2]) & set(TK[:t2])) / float(t2)
                 , len(set(X[:t2]) & set(TDD[:t2])) / float(t2)
                 , len(set(X[:t2]) & set(TDC[:t2])) / float(t2)])

    print(top5)
    print(top10)
    return top5, top10

def TDTCTB(G):
    # TD = [float(x[0]) for x in TemporalDegree(G)]
    TC = [float(x[0]) for x in TemporalCloseness(G)]
    TB = [float(x[0]) for x in TemporalBetweenness(G)]
    TK = [float(x[0]) for x in TemporalKshell(G)]
    TDC= [float(x[0]) for x in TemporalDC(G, 0.05, 1)]
    TDD = [float(x[0]) for x in TemporalDegreeDeviation(G)]
    return TC, TB, TK, TDD, TDC

def draw(x, LM, NN, TC, TB, TK, TDD, TDC, name):
    plt.plot(x, LM, 'r', label=r'$MLI_{LM}$', marker='o')
    plt.plot(x, NN, 'b', label=r'$MLI_{NN}$', marker='v')
    plt.plot(x, TC, color='g', label='TC', marker='*')
    plt.plot(x, TB, color='y', label='TB', marker='^')
    plt.plot(x, TK, color='k', label='TK', marker='>')
    plt.plot(x, TDD, color='b', label='TDD', marker='.')
    plt.plot(x, TDC, color='g', label='TDC', marker='<')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 1.1)
    plt.xlabel(r'$\beta$', fontsize=30)
    plt.ylabel(r'$HR$', fontsize=30)
    # plt.title(name,fontsize=30)

def ChangeParameter(flag):
    # Train = []
    # for i in range(30):
    #     BA = nx.read_edgelist('TemporalBA1000/BA' + str(i))
    #     Train.append(BA)
    # N_Train = 1000

    TSF = []
    for i in range(30, 60, 1):
        BA = nx.read_edgelist('TemporalBA/BA' + str(i))
        TSF.append(BA)
    N_TSF = TSF[0].number_of_nodes()
    data_TSF = np.load('data_TSF_D_0.05_1.npy')
    # TSFA, TSFD = calAD(TSF, N_TSF)

    FRI = FRINetworks()
    N_FRI = FRI[0].number_of_nodes()
    data_FRI = np.load('data_FRI_D_0.05_1.npy')

    Contact = ContactNetworks()
    N_Contact = Contact[0].number_of_nodes()
    data_Contact = np.load('data_Contact_D_0.05_1.npy')

    Hypertext = HypertextNetworks()
    N_Hypertext = Hypertext[0].number_of_nodes()
    data_Hypertext = np.load('data_Hypertext_D_0.05_1.npy')

    x = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    R_TSF = []
    R_FRI = []
    R_Contact = []
    R_Hypertext = []
    fig, ax = plt.subplots(nrows=2, ncols=2)
    for i in x:
        data_train = np.load('data_train_D_' + str(i) + '_1.npy')
        R_TSF.append(MLtest(data_train, data_TSF, N_TSF)[flag])
        R_FRI.append(MLtest(data_train, data_FRI, N_FRI)[flag])
        R_Contact.append(MLtest(data_train, data_Contact, N_Contact)[flag])
        R_Hypertext.append(MLtest(data_train, data_Hypertext, N_Hypertext)[flag])
    plt.subplot(2, 2, 1)
    plt.plot(x, R_TSF)
    plt.title('TSF')
    plt.xlabel(r'$\beta$')
    plt.ylim(0,1.1)
    plt.ylabel(r'$r$')
    plt.subplot(2, 2, 2)
    plt.plot(x, R_FRI)
    plt.title('FRI')
    plt.xlabel(r'$\beta$')
    plt.ylim(0, 1.1)
    plt.ylabel(r'$r$')
    plt.subplot(2, 2, 3)
    plt.plot(x, R_Contact)
    plt.title('Contact')
    plt.xlabel(r'$\beta$')
    plt.ylim(0, 1.1)
    plt.ylabel(r'$r$')
    plt.subplot(2, 2, 4)
    plt.plot(x, R_Hypertext)
    plt.title('Hypertext')
    plt.xlabel(r'$\beta$')
    plt.ylim(0, 1.1)
    plt.ylabel(r'$r$')
    plt.show()

def ChangeReal(flag):
    x = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    # x = [0.05]
    TSF = []
    for i in range(30):
        BA = nx.read_edgelist('TemporalBA/BA' + str(i))
        TSF.append(BA)
    N_TSF = TSF[0].number_of_nodes()
    allT_TSF = TDTCTB(TSF)

    FRI = FRINetworks()
    N_FRI = FRI[0].number_of_nodes()
    allT_FRI = TDTCTB(FRI)

    Contact = ContactNetworks()
    N_Contact = Contact[0].number_of_nodes()
    allT_Contact = TDTCTB(Contact)

    Hypertext = HypertextNetworks()
    N_Hypertext = Hypertext[0].number_of_nodes()
    allT_Hypertext = TDTCTB(Hypertext)

    # for i in x:
    #     makeDataDegree(TSF, i, 1, 'TSF')
    #     makeDataDegree(FRI, i, 1, 'FRI')
    #     makeDataDegree(Contact, i, 1, 'Contact')
    #     makeDataDegree(Hypertext, i, 1, 'Hypertext')


    LM_TSF = []
    LM_FRI = []
    LM_Contact = []
    LM_Hypertext = []

    NN_TSF = []
    NN_FRI = []
    NN_Contact = []
    NN_Hypertext = []

    TDD_TSF = []
    TDD_FRI = []
    TDD_Contact = []
    TDD_Hypertext = []

    TC_TSF = []
    TC_FRI = []
    TC_Contact = []
    TC_Hypertext = []

    TB_TSF = []
    TB_FRI = []
    TB_Contact = []
    TB_Hypertext = []

    TK_TSF = []
    TK_FRI = []
    TK_Contact = []
    TK_Hypertext = []

    TDC_TSF = []
    TDC_FRI = []
    TDC_Contact = []
    TDC_Hypertext = []

    fig, ax = plt.subplots(nrows=2, ncols=2)
    for i in x:
        print(x)
        data_train = np.load('data_train_D_0.05_1.npy')
        tempTSF = Top(data_train, np.load('data_TSF_D_' + str(i) + '_1.npy'), N_TSF, allT_TSF)[flag]
        LM_TSF.append(tempTSF[0])
        NN_TSF.append(tempTSF[1])
        TC_TSF.append(tempTSF[2])
        TB_TSF.append(tempTSF[3])
        TK_TSF.append(tempTSF[4])
        TDD_TSF.append(tempTSF[5])
        TDC_TSF.append(tempTSF[6])

        tempFRI = Top(data_train, np.load('data_FRI_D_' + str(i) + '_1.npy'), N_FRI, allT_FRI)[flag]
        LM_FRI.append(tempFRI[0])
        NN_FRI.append(tempFRI[1])
        TC_FRI.append(tempFRI[2])
        TB_FRI.append(tempFRI[3])
        TK_FRI.append(tempFRI[4])
        TDD_FRI.append(tempFRI[5])
        TDC_FRI.append(tempFRI[6])

        tempContact = Top(data_train, np.load('data_Contact_D_' + str(i) + '_1.npy'), N_Contact, allT_Contact)[flag]
        LM_Contact.append(tempContact[0])
        NN_Contact.append(tempContact[1])
        TC_Contact.append(tempContact[2])
        TB_Contact.append(tempContact[3])
        TK_Contact.append(tempContact[4])
        TDD_Contact.append(tempContact[5])
        TDC_Contact.append(tempContact[6])

        tempHypertext = Top(data_train, np.load('data_Hypertext_D_' + str(i) + '_1.npy'), N_Hypertext, allT_Hypertext)[flag]
        LM_Hypertext.append(tempHypertext[0])
        NN_Hypertext.append(tempHypertext[1])
        TC_Hypertext.append(tempHypertext[2])
        TB_Hypertext.append(tempHypertext[3])
        TK_Hypertext.append(tempHypertext[4])
        TDD_Hypertext.append(tempHypertext[5])
        TDC_Hypertext.append(tempHypertext[6])

    plt.subplot(2, 2, 1)
    draw(x, LM_TSF,NN_TSF, TC_TSF, TB_TSF, TK_TSF, TDD_TSF, TDC_TSF, 'TSF')
    plt.legend(loc='best', fontsize=11)
    plt.subplot(2, 2, 2)
    draw(x, LM_FRI, NN_FRI, TC_FRI, TB_FRI, TK_FRI, TDD_FRI, TDC_FRI, 'FRI')
    plt.subplot(2, 2, 3)
    draw(x, LM_Contact, NN_Contact, TC_Contact, TB_Contact, TK_Contact, TDD_Contact, TDC_Contact, 'Contact')
    plt.subplot(2, 2, 4)
    draw(x, LM_Hypertext, NN_Hypertext, TC_Hypertext, TB_Hypertext, TK_Hypertext, TDD_Hypertext, TDC_Hypertext, 'Hypertext')

    plt.show()

def MakeHeatMapData(flag):
    x = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    Train = []
    for i in range(30):
        BA = nx.read_edgelist('TemporalBA1000/BA' + str(i))
        Train.append(BA)
    N_Train = 1000
    # global TrainA
    # global TrainD
    # TrainA, TrainD = calAD(Train, N_Train)


    TSF = []
    for i in range(30):
        BA = nx.read_edgelist('TemporalBA/BA' + str(i))
        TSF.append(BA)
    N_TSF = TSF[0].number_of_nodes()
    # TSFA, TSFD = calAD(TSF, N_TSF)


    FRI = FRINetworks()
    N_FRI = FRI[0].number_of_nodes()
    # FRIA, FRID = calAD(FRI, N_FRI)


    Contact = ContactNetworks()
    N_Contact = Contact[0].number_of_nodes()
    # ContactA, ContactD = calAD(Contact, N_Contact)


    Hypertext = HypertextNetworks()
    N_Hypertext = Hypertext[0].number_of_nodes()
    # HypertextA, HypertextD = calAD(Hypertext, N_Hypertext)


    R_TSF = []
    R_FRI = []
    R_Contact = []
    R_Hypertext = []

    fig, ax = plt.subplots(nrows=2, ncols=2)
    for i in x:
        print(i)
        data_train = np.load('data_train_D_'+str(i)+'_1.npy')
        for j in x:
            tempTSF = DP(data_train, np.load('data_TSF_D_' + str(j) + '_1.npy'), N_TSF, 30)[flag]
            R_TSF.append(tempTSF)

            tempFRI = DP(data_train, np.load('data_FRI_D_' + str(j) + '_1.npy'), N_FRI, 30)[flag]
            R_FRI.append(tempFRI)

            tempContact = DP(data_train, np.load('data_Contact_D_' + str(j) + '_1.npy'), N_Contact, 30)[flag]
            R_Contact.append(tempContact)

            tempHypertext = DP(data_train, np.load('data_Hypertext_D_' + str(j) + '_1.npy'), N_Hypertext, 30)[flag]
            R_Hypertext.append(tempHypertext)

            # tempTSF = MLtest(data_train, np.load('data_TSF_D_' + str(j) + '_1.npy'), N_TSF)[flag]
            # R_TSF.append(tempTSF)
            #
            # tempFRI = MLtest(data_train, np.load('data_FRI_D_' + str(j) + '_1.npy'), N_FRI)[flag]
            # R_FRI.append(tempFRI)
            #
            # tempContact = MLtest(data_train, np.load('data_Contact_D_' + str(j) + '_1.npy'), N_Contact)[flag]
            # R_Contact.append(tempContact)
            #
            # tempHypertext = MLtest(data_train, np.load('data_Hypertext_D_' + str(j) + '_1.npy'), N_Hypertext)[flag]
            # R_Hypertext.append(tempHypertext)

    import pandas as pd
    import seaborn as sns
    plt.subplot(2, 2, 1)
    y = np.array(R_TSF).reshape((10, 10))
    df = pd.DataFrame(y)
    sns.heatmap(df, annot=False, vmin=0.5,vmax=1 ,xticklabels=x, yticklabels=x)
    plt.title('TSF', fontsize=20)
    plt.xlabel(r'$\beta$', fontsize=20)
    plt.ylabel(r'$\beta_{t}$', fontsize=20)

    plt.subplot(2, 2, 2)
    y = np.array(R_FRI).reshape((10, 10))
    df = pd.DataFrame(y)
    sns.heatmap(df, annot=False, vmin=0.5, vmax=1, xticklabels=x, yticklabels=x)
    plt.title('FRI', fontsize=20)
    plt.xlabel(r'$\beta$', fontsize=20)
    plt.ylabel(r'$\beta_{t}$', fontsize=20)

    plt.subplot(2, 2, 3)
    y = np.array(R_Contact).reshape((10, 10))
    df = pd.DataFrame(y)
    sns.heatmap(df, annot=False, vmin=0.5, vmax=1, xticklabels=x, yticklabels=x)
    plt.title('Contact', fontsize=20)
    plt.xlabel(r'$\beta$', fontsize=20)
    plt.ylabel(r'$\beta_{t}$', fontsize=20)

    plt.subplot(2, 2, 4)
    y = np.array(R_Hypertext).reshape((10, 10))
    df = pd.DataFrame(y)
    sns.heatmap(df, annot=False, vmin=0.5, vmax=1, xticklabels=x, yticklabels=x)
    plt.title('Hypertext', fontsize=20)
    plt.xlabel(r'$\beta$', fontsize=20)
    plt.ylabel(r'$\beta_{t}$', fontsize=20)

    plt.show()



if __name__ == '__main__':
    # X = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    # for x in X:
    #     TSF = []
    #     for i in range(30):
    #         BA = nx.read_edgelist('TemporalBA1000/BA' + str(i))
    #         TSF.append(BA)
    #     N = TSF[0].number_of_nodes()
    #     makeDataAll(TSF, x, 1, 'train')
    #
    #
    #
    #     TSF = []
    #     for i in range(30):
    #         BA = nx.read_edgelist('TemporalBA/BA' + str(i))
    #         TSF.append(BA)
    #     N_TSF = TSF[0].number_of_nodes()
    #     makeDataAll(TSF, x, 1, 'TSF')
    # # data_TSF = np.load('data_TSF_D_0.05_1.npy')
    #
    #     FRI = FRINetworks()
    #     N_FRI = FRI[0].number_of_nodes()
    #     makeDataAll(FRI, x, 1, 'FRI')
    # # data_FRI = np.load('data_FRI_D_0.05_1.npy')
    #
    #     Contact = ContactNetworks()
    #     N_Contact = Contact[0].number_of_nodes()
    #     makeDataAll(Contact, x, 1, 'Contact')
    # # data_Contact = np.load('data_Contact_D_0.05_1.npy')
    #
    #     Hypertext = HypertextNetworks()
    #     N_Hypertext = Hypertext[0].number_of_nodes()
    #     makeDataAll(Hypertext, x, 1, 'Hypertext')
    # data_Hypertext = np.load('data_Hypertext_D_0.05_1.npy')

    # Infectious = InfectiousNetworks()
    # N = Infectious[0].number_of_nodes()
    # makeDataDegree(Infectious, 0.05, 1, 'Infectious')
    # data_Infectious = np.load('data_Infectious_D_0.05_1.npy')

    # DNC = DNCNetworks()
    # N = DNC[0].number_of_nodes()
    # makeDataDegree(DNC, 0.01, 0.1, 'DNC')
    # data_DNC = np.load('data_DNC_D_0.01_0.1.npy')
    #


    # data_train = np.load('data_train_D_0.05_1.npy')
    # Top(Infectious, data_train, data_Infectious, N)

    # ChangeParameter(1)
    # ChangeReal(1)
    # MakeHeatMapData(1)
    TSF = []
    for i in range(30):
        BA = nx.read_edgelist('TemporalBA/BA' + str(i))
        TSF.append(BA)
    N_TSF = TSF[0].number_of_nodes()

    t1 = int(N_TSF * 0.05)
    t2 = int(N_TSF * 0.1)
    data_test = np.load('data_TSF_D_0.05_1.npy')
    X = data_test[data_test[:, -1].argsort()][::-1, :][:, 0]
    TDC_TSF = [float(x[0]) for x in TemporalDC(TSF, 0.05, 1)]
    print(len(set(X[:t1]) & set(TDC_TSF[:t1])) / float(t1))
    print(len(set(X[:t2]) & set(TDC_TSF[:t2])) / float(t2))


    FRI = FRINetworks()
    N_FRI = FRI[0].number_of_nodes()
    t1 = int(N_FRI * 0.05)
    t2 = int(N_FRI * 0.1)
    data_test = np.load('data_FRI_D_0.05_1.npy')
    X = data_test[data_test[:, -1].argsort()][::-1, :][:, 0]
    TDC_FRI = [float(x[0]) for x in TemporalDC(FRI, 0.05, 1)]
    print(len(set(X[:t1]) & set(TDC_FRI[:t1])) / float(t1))
    print(len(set(X[:t2]) & set(TDC_FRI[:t2])) / float(t2))


    Contact = ContactNetworks()
    N_Contact = Contact[0].number_of_nodes()
    t1 = int(N_Contact * 0.05)
    t2 = int(N_Contact * 0.1)
    data_test = np.load('data_Contact_D_0.05_1.npy')
    X = data_test[data_test[:, -1].argsort()][::-1, :][:, 0]
    TDC_Contact = [float(x[0]) for x in TemporalDC(Contact, 0.05, 1)]
    print(len(set(X[:t1]) & set(TDC_Contact[:t1])) / float(t1))
    print(len(set(X[:t2]) & set(TDC_Contact[:t2])) / float(t2))

    Hypertext = HypertextNetworks()
    N_Hypertext = Hypertext[0].number_of_nodes()
    t1 = int(N_Hypertext * 0.05)
    t2 = int(N_Hypertext * 0.1)
    data_test = np.load('data_Hypertext_D_0.05_1.npy')
    X = data_test[data_test[:, -1].argsort()][::-1, :][:, 0]
    TDC_Hypertext = [float(x[0]) for x in TemporalDC(Hypertext, 0.05, 1)]
    print(len(set(X[:t1]) & set(TDC_Hypertext[:t1])) / float(t1))
    print(len(set(X[:t2]) & set(TDC_Hypertext[:t2])) / float(t2))





    # FRI = chooseNetworks(FRINetworks(),10)
    # FRI = chooseNetworks(FRI,10)
    # allSIR = ALLSIR(FRI, 0.01, 0.1)
    # re=[]
    # for node in TDC(FRI, 0.01, 0.1):
    #     re.append(allSIR[node[0]])
    #
    # print 'TDC', kendall(range(len(re)), re)
    #
    # re = []
    # for node in TemporalDegree(FRI):
    #     re.append(allSIR[node[0]])
    #
    # print 'Degree', kendall(range(len(re)), re)
    #
    # re = []
    # for node in TemporalBetweenness(FRI):
    #     re.append(allSIR[node[0]])
    #
    # print 'Betweenness', kendall(range(len(re)), re)
    #
    # re = []
    # for node in TemporalCloseness(FRI):
    #     re.append(allSIR[node[0]])
    #
    # print 'Closeness', kendall(range(len(re)), re)