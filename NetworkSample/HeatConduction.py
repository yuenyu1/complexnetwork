#coding=utf-8
#P与最终结果的关系,0
#抽样率
#不同的网络

import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import time

def constructG(filename):
    G = nx.DiGraph().to_undirected()
    G.name = filename
    with open('/Users/yuenyu/PycharmProjects/complexnetwork/networks/'+filename+'', 'r') as f:
        for position, line in enumerate(f):
            t = line.strip().split(' ')
            if t[0] != t[1]:
                G.add_edge(t[0], t[1])
    return G

def heatConduction(G, frac, u):
    nodes = G.nodes()
    infected = [random.choice(nodes)]#初始为1个感染节点
    flag = {}
    for node in nodes:
        flag[node] = 0
    flag[infected[0]] = 1
    N = 0
    while True:
        N += 1
        if N > 6*G.number_of_nodes():
            infected = [random.choice(nodes)] # 初始为1个感染节点
            flag = {}
            for node in nodes:
                flag[node] = 0
            flag[infected[0]] = 1
            N = 0
            pass

        select = random.choice(infected)
        temp = 0
        for neighbor in G.neighbors(select):
            temp += max(0.01, 1.0/(G.degree(neighbor)**u))
        r = random.uniform(0, 1)
        temp2 = 0
        for neighbor in G.neighbors(select):
            temp2 += max(0.01, 1.0/(G.degree(neighbor)**u))
            if temp2/temp >= r:
                if flag[neighbor] == 0:
                    infected.append(neighbor)
                    flag[neighbor] = 1
                break

        if len(infected) >= frac*G.number_of_nodes():
            return infected
    return infected

def my2(G, frac, u):
    nodes = G.nodes()
    infected = random.choice(nodes)#初始为1个感染节点
    reNodes = []
    reNodes.append(infected)
    flag = {}
    for node in nodes:
        flag[node] = 0
    flag[infected] = 1
    N = 0
    while True:
        N += 1
        if N > 6*G.number_of_nodes():
            infected = random.choice(nodes)
            reNodes = []
            reNodes.append(infected)
            flag = {}
            for node in nodes:
                flag[node] = 0
            flag[infected] = 1
            N = 0
            pass

        temp = 0
        for neighbor in G.neighbors(infected):
            temp += max(0.01, G.degree(neighbor)**u)
        r = random.uniform(0, 1)
        temp2 = 0
        for neighbor in G.neighbors(infected):
            temp2 += max(0.01, G.degree(neighbor)**u)
            if temp2/float(temp) >= r:
                infected = neighbor
                if flag[neighbor] == 0:
                    reNodes.append(neighbor)
                    flag[neighbor] = 1

                break

        if len(reNodes) >= frac*G.number_of_nodes():
            return reNodes
    return reNodes

def my3(G, frac, p):
    nodes = G.nodes()
    infected = [random.choice(nodes)]  # 初始为1个感染节点
    flag = {}
    for node in nodes:
        flag[node] = 0
    flag[infected[0]] = 1
    N = 0
    while True:
        N += 1
        if N > 6 * G.number_of_nodes():
            infected = [random.choice(nodes)]  # 初始为1个感染节点
            flag = {}
            for node in nodes:
                flag[node] = 0
            flag[infected[0]] = 1
            N = 0
            pass
        if infected == []:
            infected = [random.choice(nodes)]
        select = random.choice(infected)
        temp = random.uniform(0, 1)
        if temp < p:
            infected.remove(select)
            flag[select] = 0
        else:
            tempNode = random.choice(G.neighbors(select))
            if flag[tempNode] == 0:
                infected.append(tempNode)
                flag[tempNode] = 1

        if len(infected) >= frac * G.number_of_nodes():
            return infected
    return infected

def mySample(G, frac, p):
    while True:
        infectedNodes = heatConduction(G, p, frac)
        if infectedNodes != []:
            sampleG = nx.subgraph(G, infectedNodes)
            return sampleG


def RW(G, fracs):
    nodes = G.nodes()
    infected = random.choice(nodes)  # 初始为1个感染节点
    reNodes = []
    Edges = []
    reEdges = []
    reNodes.append(infected)
    flag = {}
    for node in nodes:
        flag[node] = 0
    flag[infected] = 1
    N = 0
    f = 0
    while True:
        N += 1
        if N > 6 * G.number_of_nodes():
            infected = random.choice(nodes)  # 初始为1个感染节点
            reNodes = []
            Edges = []
            reEdges = []
            reNodes.append(infected)
            flag = {}
            for node in nodes:
                flag[node] = 0
            flag[infected] = 1
            N = 0
            f = 0
        tempNode = random.choice(G.neighbors(infected))
        Edges.append((infected, tempNode))
        infected = tempNode
        if flag[infected] == 0:
            flag[infected] = 1
            reNodes.append(infected)
        else:
            pass

        if len(reNodes) >= fracs[f]*G.number_of_nodes():
            reEdges.append(Edges[:])
            f = f+1
            if f >= len(fracs):
                return reEdges
    return reEdges

def RWI(G, frac):
    nodes = G.nodes()
    infected = random.choice(nodes)  # 初始为1个感染节点
    reNodes = []
    reNodes.append(infected)
    flag = {}
    for node in nodes:
        flag[node] = 0
    flag[infected] = 1
    N = 0
    while True:
        N += 1
        if N > 6*G.number_of_nodes():
            infected = random.choice(nodes)  # 初始为1个感染节点
            reNodes = []
            reNodes.append(infected)
            flag = {}
            for node in nodes:
                flag[node] = 0
            flag[infected] = 1
            N = 0
        tempNode = random.choice(G.neighbors(infected))
        infected = tempNode
        if flag[infected] == 0:
            reNodes.append(infected)
            flag[infected] = 1
        if len(reNodes) >= frac*G.number_of_nodes():
            return reNodes
    return reNodes

def RWIWithReelect(G, frac):
    nodes = G.nodes()
    infected = [random.choice(nodes)]  # 初始为1个感染节点
    flag = {}
    for node in nodes:
        flag[node] = 0
    flag[infected[0]] = 1
    N = 0
    while True:
        N += 1
        if N > 6 * G.number_of_nodes():
            infected = [random.choice(nodes)]  # 初始为1个感染节点
            flag = {}
            for node in nodes:
                flag[node] = 0
            flag[infected[0]] = 1
            N = 0
            pass

        select = random.choice(infected)
        temp = random.choice(G.neighbors(select))
        if flag[temp] == 0:
            infected.append(temp)
            flag[temp] = 1

        if len(infected) >= frac * G.number_of_nodes():
            return infected
    return infected

def MHRW(G, fracs):
    nodes = G.nodes()
    infected = random.choice(nodes)  # 初始为1个感染节点
    reNodes = []
    Edges = []
    reEdges = []
    reNodes.append(infected)
    flag = {}
    for node in nodes:
        flag[node] = 0
    flag[infected] = 1
    N = 0
    f = 0
    while True:
        N += 1
        if N > 6 * G.number_of_nodes():
            infected = random.choice(nodes)  # 初始为1个感染节点
            reNodes = []
            Edges = []
            reEdges = []
            reNodes.append(infected)
            flag = {}
            for node in nodes:
                flag[node] = 0
            flag[infected] = 1
            N = 0
            f = 0
        tempNode = random.choice(G.neighbors(infected))
        p = random.uniform(0, 1)
        if p < max(0.01, G.degree(infected)/float(G.degree(tempNode))):
            Edges.append((infected, tempNode))
            infected = tempNode
            if flag[infected] == 0:
                flag[infected] = 1
                reNodes.append(infected)
        else:
            pass

        if len(reNodes) >= fracs[f]*G.number_of_nodes():
            reEdges.append(Edges[:])
            f = f+1
            if f >= len(fracs):
                return reEdges
    return reEdges

def MHRWI(G, frac):
    nodes = G.nodes()
    infected = random.choice(nodes)  # 初始为1个感染节点
    reNodes = []
    reNodes.append(infected)
    flag = {}
    for node in nodes:
        flag[node] = 0
    flag[infected] = 1
    N = 0

    while True:
        N += 1
        if N > 6 * G.number_of_nodes():
            infected = random.choice(nodes)  # 初始为1个感染节点
            reNodes = []
            reNodes.append(infected)
            flag = {}
            for node in nodes:
                flag[node] = 0
            flag[infected] = 1
            N = 0

        tempNode = random.choice(G.neighbors(infected))
        p = random.uniform(0, 1)
        if p < max(0.01, G.degree(infected)/float(G.degree(tempNode))):
            infected = tempNode
            if flag[infected] == 0:
                flag[infected] = 1
                reNodes.append(infected)
        else:
            pass

        if len(reNodes) >= frac*G.number_of_nodes():
            return reNodes
    return reNodes

def MHRWWithReelect(G, frac):
    nodes = G.nodes()
    infected = [random.choice(nodes)]  # 初始为1个感染节点
    flag = {}
    for node in nodes:
        flag[node] = 0
    flag[infected[0]] = 1
    N = 0
    while True:
        N += 1
        if N > 6 * G.number_of_nodes():
            infected = [random.choice(nodes)]  # 初始为1个感染节点
            flag = {}
            for node in nodes:
                flag[node] = 0
            flag[infected[0]] = 1
            N = 0
        tempNode = random.choice(infected)
        neighbor = random.choice(G.neighbors(tempNode))
        p = random.uniform(0, 1)
        if p < max(0.01, G.degree(tempNode)/float(G.degree(neighbor))):
            if flag[neighbor] == 0:
                infected.append(neighbor)
                flag[neighbor] = 1
        else:
            pass
        if len(infected) >= frac*G.number_of_nodes():
            return infected
    return infected


def SST(G):
    g = G.copy()
    for edge in g.edges():
        g.add_edge(edge[0], edge[1], score=0)
    N = 50
    nodes = G.nodes()
    reNodes = []
    flag = {}
    for node in nodes:
        flag[node] = 0
    while N > 0:
        N -= 1
        infected = random.choice(nodes)  # 初始为1个感染节点
        for edge in nx.dfs_edges(G, infected):
            g.add_edge(edge[0], edge[1], score=g.edge[edge[0]][edge[1]]['score'] + 1)
    edgesDict = nx.get_edge_attributes(g, 'score')
    edgeRank = sorted(edgesDict.iteritems(), key=lambda x: x[1], reverse=True)
    for edge in edgeRank:
        if flag[edge[0][0]] == 0:
            flag[edge[0][0]] = 1
            reNodes.add(edge[0][0])
        if flag[edge[0][1]] == 0:
            flag[edge[0][1]] = 1
            reNodes.add(edge[0][1])
    return reNodes

def BFS(G, frac):
    L = {}
    Discovered = {}
    for node in G.nodes():
        Discovered[node] = False
    s = random.choice(G.nodes())
    Discovered[s] = True
    BFSNode = [s]
    i = 0 #计算器
    L[i] = [s]
    Length = 1
    while L[i] != []:
        L[i+1] = []
        for node in L[i]:
            for neighbor in G.neighbors(node):
                if Discovered[neighbor] == False:
                    Discovered[neighbor] = True
                    L[i+1].append(neighbor)
                    BFSNode.append(neighbor)
                    Length += 1
                    if Length >= frac*G.number_of_nodes():
                        return BFSNode
        i += 1
    return BFSNode

def FFS(G, fracs):
    L = {}
    Discovered = {}
    for node in G.nodes():
        Discovered[node] = False
    s = random.choice(G.nodes())
    Discovered[s] = True
    FFSEdges = []
    reEdges = []
    i = 0  # 计算器
    L[i] = set([s])
    Length = 1
    f = 0
    N = 0
    while len(L[i])> 0:
        N +=1
        if N > 6 * G.number_of_nodes():
            L = {}
            Discovered = {}
            for node in G.nodes():
                Discovered[node] = False
            s = random.choice(G.nodes())
            Discovered[s] = True
            FFSEdges = []
            reEdges = []
            i = 0  # 计算器
            L[i] = set([s])
            Length = 1
            f = 0
            N = 0
        L[i + 1] = set()
        for node in L[i]:
            from scipy.stats import geom
            numNeighbors = int(geom.rvs(0.7))
            tempArray = G.neighbors(node)
            random.shuffle(tempArray)
            for neighbor in tempArray[:numNeighbors]:
                FFSEdges.append((node, neighbor))
                L[i + 1].add(neighbor)
                if Discovered[neighbor] == False:
                    Discovered[neighbor] = True
                    Length += 1
                    if Length >= fracs[f] * G.number_of_nodes():
                        reEdges.append(FFSEdges[:])
                        f += 1
                        if f >= len(fracs):
                            return reEdges
        i += 1
    return reEdges

def FFI(G, frac):
    L = {}
    Discovered = {}
    for node in G.nodes():
        Discovered[node] = False
    s = random.choice(G.nodes())
    Discovered[s] = True
    reNodes = []
    i = 0  # 计算器
    L[i] = set([s])
    Length = 1
    N = 0
    while len(L[i])> 0:
        N +=1
        if N > 6 * G.number_of_nodes():
            L = {}
            Discovered = {}
            for node in G.nodes():
                Discovered[node] = False
            s = random.choice(G.nodes())
            Discovered[s] = True
            reNodes = []
            i = 0  # 计算器
            L[i] = set([s])
            Length = 1
            N = 0
        L[i + 1] = set()
        for node in L[i]:
            from scipy.stats import geom
            numNeighbors = int(geom.rvs(0.7))
            tempArray = G.neighbors(node)
            random.shuffle(tempArray)
            for neighbor in tempArray[:numNeighbors]:
                L[i + 1].add(neighbor)
                if Discovered[neighbor] == False:
                    Discovered[neighbor] = True
                    reNodes.append(neighbor)
                    Length += 1
                    if Length >= frac * G.number_of_nodes():
                        return reNodes
        i += 1
    return reNodes

def RND(G, frac):
    degree = G.degree()
    nodes = []
    for node in degree.keys():
        for i in range(degree[node]):
            nodes.append(node)
    reNodes = []
    flag = {}
    for node in G.nodes():
        flag[node] = 0
    while True:
        tempNode = random.choice(nodes)
        if flag[tempNode] == 0:
            flag[tempNode] = 1
            reNodes.append(tempNode)
        if len(reNodes) >= frac * G.number_of_nodes():
            return reNodes


def CCD(G):
    if G.number_of_nodes() == 0:
        return []
    re = [0]*101
    for i in nx.clustering(G).values():
        re[int(i*100)] += 1
    temp = [z/float(sum(re)) for z in re]
    # print temp
    return temp

def CCD2(G):
    if G.number_of_nodes() == 0:
        return []
    re = [0]*101
    for i in nx.clustering(G).values():
        re[int(i*100)] += 1
    s = 0
    temp = []
    all = float(sum(re))
    for i in re:
        s += i
        temp.append(s/all)
    return temp

def DD(G):
    if G.number_of_nodes() == 0:
        return []
    x = nx.degree_histogram(G)
    return [z/float(sum(x)) for z in x]

def DD2(G):
    if G.number_of_nodes() == 0:
        return []
    x = nx.degree_histogram(G)
    s = 0
    temp = []
    all = float(sum(x))
    for i in x:
        s += i
        temp.append(s / all)
    return temp

def KSD(d1, d2):
    d2 = d2 + (len(d1)-len(d2))*[0]
    maxD = 0
    for i in range(len(d1)):
        if abs(d1[i] - d2[i]) > maxD:
            maxD = abs(d1[i] - d2[i])
    return maxD

def ND(d1, d2):
    d2 = d2 + (len(d1) - len(d2)) * [0]
    return math.sqrt(np.sum(np.square(np.array(d1)-np.array(d2))))/math.sqrt(float(np.sum(np.square(np.array(d1)))))

def SDD(d1, d2):
    d2 = d2 + (len(d1) - len(d2)) * [1]
    P = 0.3*np.array(d1)+(1-0.3)*np.array(d2)
    Q = 0.3*np.array(d2)+(1-0.3)*np.array(d1)
    re = 0
    for i in range(len(P)):
        if P[i] != 0 and Q[i] != 0:
            re += math.log(P[i]/Q[i])*P[i]
    return re

def networksInfo(networks):
    for network in networks:
        print 'name:' + network.name
        print 'nodes:' + str(network.number_of_nodes())
        print 'edges:' + str(network.number_of_edges())
        print 'average degree:' + str(2*network.number_of_edges()/float(network.number_of_nodes()))
        print 'clustering coefficient:' + str(nx.average_clustering(network))


def caculateMySample(G, fracs, u):
    KSD_DD = [0] * len(fracs)
    KSD_CCD = [0] * len(fracs)
    # ND_DD = [0] * len(fracs)
    # ND_CCD = [0] * len(fracs)
    # SDD_DD = [0] * len(fracs)
    # SDD_CCD = [0] * len(fracs)
    for i in range(30):
        print i
        reNodes = heatConduction(G, fracs[-1], u)
        for j in range(len(fracs)):
            tempNodes = reNodes[:int(fracs[j] * G.number_of_nodes())]
            sampleG = nx.subgraph(G, tempNodes)
            dd2 = DD(sampleG)
            ccd2 = CCD(sampleG)
            KSD_DD[j] += KSD(dd1, dd2)
            KSD_CCD[j] += KSD(ccd1, ccd2)
            # ND_DD[j] += ND(dd1, dd2)
            # ND_CCD[j] += ND(ccd1, ccd2)
            # SDD_DD[j] += SDD(dd1, dd2)
            # SDD_CCD[j] += SDD(ccd1, ccd2)
    return ([x / 30.0 for x in KSD_DD], [x / 30.0 for x in KSD_CCD])

def caculateMySample2(G, fracs, u):
    KSD_DD = [0] * len(fracs)
    KSD_CCD = [0] * len(fracs)
    # ND_DD = [0] * len(fracs)
    # ND_CCD = [0] * len(fracs)
    # SDD_DD = [0] * len(fracs)
    # SDD_CCD = [0] * len(fracs)
    for i in range(30):
        print i
        reNodes = my2(G, fracs[-1], u)
        for j in range(len(fracs)):
            tempNodes = reNodes[:int(fracs[j] * G.number_of_nodes())]
            sampleG = nx.subgraph(G, tempNodes)
            dd2 = DD(sampleG)
            ccd2 = CCD(sampleG)
            KSD_DD[j] += KSD(dd1, dd2)
            KSD_CCD[j] += KSD(ccd1, ccd2)
            # ND_DD[j] += ND(dd1, dd2)
            # ND_CCD[j] += ND(ccd1, ccd2)
            # SDD_DD[j] += SDD(dd1, dd2)
            # SDD_CCD[j] += SDD(ccd1, ccd2)
    return ([x / 30.0 for x in KSD_DD], [x / 30.0 for x in KSD_CCD])

def caculateMySample3(G, fracs, p):
    KSD_DD = [0] * len(fracs)
    KSD_CCD = [0] * len(fracs)
    ND_DD = [0] * len(fracs)
    ND_CCD = [0] * len(fracs)
    # SDD_DD = [0] * len(fracs)
    # SDD_CCD = [0] * len(fracs)
    for i in range(30):
        print i
        reNodes = my3(G, fracs[-1], p)
        for j in range(len(fracs)):
            tempNodes = reNodes[:int(fracs[j] * G.number_of_nodes())]
            sampleG = nx.subgraph(G, tempNodes)
            dd2 = DD(sampleG)
            ccd2 = CCD(sampleG)
            KSD_DD[j] += KSD(dd1, dd2)
            KSD_CCD[j] += KSD(ccd1, ccd2)
            ND_DD[j] += ND(dd1, dd2)
            ND_CCD[j] += ND(ccd1, ccd2)
            # SDD_DD[j] += SDD(dd1, dd2)
            # SDD_CCD[j] += SDD(ccd1, ccd2)
    return ([x / 30.0 for x in KSD_DD], [x / 30.0 for x in KSD_CCD], [x / 30.0 for x in ND_DD],
            [x / 30.0 for x in ND_CCD])

def caculateRW(G, fracs):
    KSD_DD = [0] * len(fracs)
    KSD_CCD = [0] * len(fracs)
    # ND_DD = [0] * len(fracs)
    # ND_CCD = [0] * len(fracs)
    # SDD_DD = [0] * len(fracs)
    # SDD_CCD = [0] * len(fracs)
    for i in range(30):
        print i
        reEdges = RW(G, fracs)
        for j in range(len(fracs)):
            tempEdges = reEdges[j]
            sampleG = nx.DiGraph().to_undirected()
            sampleG.add_edges_from(tempEdges)

            dd2 = DD(sampleG)
            ccd2 = CCD(sampleG)
            KSD_DD[j] += KSD(dd1, dd2)
            KSD_CCD[j] += KSD(ccd1, ccd2)
            # ND_DD[j] += ND(dd1, dd2)
            # ND_CCD[j] += ND(ccd1, ccd2)
            # SDD_DD[j] += SDD(dd1, dd2)
            # SDD_CCD[j] += SDD(ccd1, ccd2)
    return ([x / 30.0 for x in KSD_DD], [x / 30.0 for x in KSD_CCD])

def caculateMHRW(G, fracs):
    KSD_DD = [0] * len(fracs)
    KSD_CCD = [0] * len(fracs)
    # ND_DD = [0] * len(fracs)
    # ND_CCD = [0] * len(fracs)
    # SDD_DD = [0] * len(fracs)
    # SDD_CCD = [0] * len(fracs)
    for i in range(30):
        print i
        reEdges = MHRW(G, fracs)
        for j in range(len(fracs)):
            tempEdges = reEdges[j]
            sampleG = nx.DiGraph().to_undirected()
            sampleG.add_edges_from(tempEdges)

            dd2 = DD(sampleG)
            ccd2 = CCD(sampleG)
            KSD_DD[j] += KSD(dd1, dd2)
            KSD_CCD[j] += KSD(ccd1, ccd2)
            # ND_DD[j] += ND(dd1, dd2)
            # ND_CCD[j] += ND(ccd1, ccd2)
            # SDD_DD[j] += SDD(dd1, dd2)
            # SDD_CCD[j] += SDD(ccd1, ccd2)
    return ([x / 30.0 for x in KSD_DD], [x / 30.0 for x in KSD_CCD])

def caculateMHRW2(G, fracs):
    KSD_DD = [0] * len(fracs)
    KSD_CCD = [0] * len(fracs)
    ND_DD = [0] * len(fracs)
    ND_CCD = [0] * len(fracs)
    # SDD_DD = [0] * len(fracs)
    # SDD_CCD = [0] * len(fracs)
    for i in range(30):
        print i
        reNodes = MHRWWithReelect(G, fracs[-1])
        for j in range(len(fracs)):
            tempNodes = reNodes[:int(fracs[j] * G.number_of_nodes())]
            sampleG = nx.subgraph(G, tempNodes)
            dd2 = DD(sampleG)
            ccd2 = CCD(sampleG)
            KSD_DD[j] += KSD(dd1, dd2)
            KSD_CCD[j] += KSD(ccd1, ccd2)
            ND_DD[j] += ND(dd1, dd2)
            ND_CCD[j] += ND(ccd1, ccd2)
            # SDD_DD[j] += SDD(dd1, dd2)
            # SDD_CCD[j] += SDD(ccd1, ccd2)
    return ([x / 30.0 for x in KSD_DD], [x / 30.0 for x in KSD_CCD], [x / 30.0 for x in ND_DD],
            [x / 30.0 for x in ND_CCD])

def caculateRWI(G,fracs):
    KSD_DD = [0]*len(fracs)
    KSD_CCD = [0]*len(fracs)
    # ND_DD = [0]*len(fracs)
    # ND_CCD = [0]*len(fracs)
    # SDD_DD = [0]*len(fracs)
    # SDD_CCD = [0]*len(fracs)
    for i in range(30):
        print i
        reNodes = RWI(G, fracs[-1])
        for j in range(len(fracs)):
            tempNodes = reNodes[:int(fracs[j] * G.number_of_nodes())]
            sampleG = nx.subgraph(G, tempNodes)
            dd2 = DD(sampleG)
            ccd2 = CCD(sampleG)
            KSD_DD[j] += KSD(dd1, dd2)
            KSD_CCD[j] += KSD(ccd1, ccd2)
            # ND_DD[j] += ND(dd1, dd2)
            # ND_CCD[j] += ND(ccd1, ccd2)
            # SDD_DD[j] += SDD(dd1, dd2)
            # SDD_CCD[j] += SDD(ccd1, ccd2)
    return ([x / 30.0 for x in KSD_DD], [x / 30.0 for x in KSD_CCD])

def caculateMHRWI(G,fracs):
    KSD_DD = [0]*len(fracs)
    KSD_CCD = [0]*len(fracs)
    # ND_DD = [0]*len(fracs)
    # ND_CCD = [0]*len(fracs)
    # SDD_DD = [0]*len(fracs)
    # SDD_CCD = [0]*len(fracs)
    for i in range(30):
        print i
        reNodes = MHRWI(G, fracs[-1])
        for j in range(len(fracs)):
            tempNodes = reNodes[:int(fracs[j] * G.number_of_nodes())]
            sampleG = nx.subgraph(G, tempNodes)
            dd2 = DD(sampleG)
            ccd2 = CCD(sampleG)
            KSD_DD[j] += KSD(dd1, dd2)
            KSD_CCD[j] += KSD(ccd1, ccd2)
            # ND_DD[j] += ND(dd1, dd2)
            # ND_CCD[j] += ND(ccd1, ccd2)
            # SDD_DD[j] += SDD(dd1, dd2)
            # SDD_CCD[j] += SDD(ccd1, ccd2)
    return ([x / 30.0 for x in KSD_DD], [x / 30.0 for x in KSD_CCD])

def caculateFFI(G,fracs):
    KSD_DD = [0]*len(fracs)
    KSD_CCD = [0]*len(fracs)
    # ND_DD = [0]*len(fracs)
    # ND_CCD = [0]*len(fracs)
    # SDD_DD = [0]*len(fracs)
    # SDD_CCD = [0]*len(fracs)
    for i in range(30):
        print i
        reNodes = FFI(G, fracs[-1])
        for j in range(len(fracs)):
            tempNodes = reNodes[:int(fracs[j] * G.number_of_nodes())]
            sampleG = nx.subgraph(G, tempNodes)
            dd2 = DD(sampleG)
            ccd2 = CCD(sampleG)
            KSD_DD[j] += KSD(dd1, dd2)
            KSD_CCD[j] += KSD(ccd1, ccd2)
            # ND_DD[j] += ND(dd1, dd2)
            # ND_CCD[j] += ND(ccd1, ccd2)
            # SDD_DD[j] += SDD(dd1, dd2)
            # SDD_CCD[j] += SDD(ccd1, ccd2)
    return ([x / 30.0 for x in KSD_DD], [x / 30.0 for x in KSD_CCD])

def caculateRWI2(G,fracs):
    KSD_DD = [0]*len(fracs)
    KSD_CCD = [0]*len(fracs)
    # ND_DD = [0]*len(fracs)
    # ND_CCD = [0]*len(fracs)
    # SDD_DD = [0]*len(fracs)
    # SDD_CCD = [0]*len(fracs)
    for i in range(30):
        print i
        reNodes = RWIWithReelect(G, fracs[-1])
        for j in range(len(fracs)):
            tempNodes = reNodes[:int(fracs[j] * G.number_of_nodes())]
            sampleG = nx.subgraph(G, tempNodes)
            dd2 = DD(sampleG)
            ccd2 = CCD(sampleG)
            KSD_DD[j] += KSD(dd1, dd2)
            KSD_CCD[j] += KSD(ccd1, ccd2)
            # ND_DD[j] += ND(dd1, dd2)
            # ND_CCD[j] += ND(ccd1, ccd2)
            # SDD_DD[j] += SDD(dd1, dd2)
            # SDD_CCD[j] += SDD(ccd1, ccd2)
    return ([x / 30.0 for x in KSD_DD], [x / 30.0 for x in KSD_CCD])

def caculateBFS(G,fracs):
    KSD_DD = [0]*len(fracs)
    KSD_CCD = [0]*len(fracs)
    # ND_DD = [0]*len(fracs)
    # ND_CCD = [0]*len(fracs)
    # SDD_DD = [0]*len(fracs)
    # SDD_CCD = [0]*len(fracs)
    for i in range(30):
        print i
        reNodes = BFS(G, fracs[-1])
        for j in range(len(fracs)):
            tempNodes = reNodes[:int(fracs[j] * G.number_of_nodes())]
            sampleG = nx.subgraph(G, tempNodes)
            dd2 = DD(sampleG)
            ccd2 = CCD(sampleG)
            KSD_DD[j] += KSD(dd1, dd2)
            KSD_CCD[j] += KSD(ccd1, ccd2)
            # ND_DD[j] += ND(dd1, dd2)
            # ND_CCD[j] += ND(ccd1, ccd2)
            # SDD_DD[j] += SDD(dd1, dd2)
            # SDD_CCD[j] += SDD(ccd1, ccd2)
    return ([x / 30.0 for x in KSD_DD], [x / 30.0 for x in KSD_CCD])

def caculateFFS(G, fracs):
    KSD_DD = [0] * len(fracs)
    KSD_CCD = [0] * len(fracs)
    # ND_DD = [0] * len(fracs)
    # ND_CCD = [0] * len(fracs)
    # SDD_DD = [0] * len(fracs)
    # SDD_CCD = [0] * len(fracs)
    for i in range(30):
        reEdges = FFS(G, fracs)
        print i
        for j in range(len(fracs)):
            tempEdges = reEdges[j]
            sampleG = nx.DiGraph().to_undirected()
            sampleG.add_edges_from(tempEdges)

            dd2 = DD(sampleG)
            ccd2 = CCD(sampleG)
            KSD_DD[j] += KSD(dd1, dd2)
            KSD_CCD[j] += KSD(ccd1, ccd2)
            # ND_DD[j] += ND(dd1, dd2)
            # ND_CCD[j] += ND(ccd1, ccd2)
            # SDD_DD[j] += SDD(dd1, dd2)
            # SDD_CCD[j] += SDD(ccd1, ccd2)
    return ([x / 30.0 for x in KSD_DD], [x / 30.0 for x in KSD_CCD])

def caculateRND(G,fracs):
    KSD_DD = [0]*len(fracs)
    KSD_CCD = [0]*len(fracs)
    # ND_DD = [0]*len(fracs)
    # ND_CCD = [0]*len(fracs)
    # SDD_DD = [0]*len(fracs)
    # SDD_CCD = [0]*len(fracs)
    for i in range(30):
        print i
        reNodes = RND(G, fracs[-1])
        for j in range(len(fracs)):
            tempNodes = reNodes[:int(fracs[j] * G.number_of_nodes())]
            # for node in reNodes:
            #     if len(tempNodes) >= fracs[j] * G.number_of_nodes():
            #         break
            #     else:
            #         tempNodes.append(node)
            sampleG = nx.subgraph(G, tempNodes)
            dd2 = DD(sampleG)
            ccd2 = CCD(sampleG)
            KSD_DD[j] += KSD(dd1, dd2)
            KSD_CCD[j] += KSD(ccd1, ccd2)
            # ND_DD[j] += ND(dd1, dd2)
            # ND_CCD[j] += ND(ccd1, ccd2)
            # SDD_DD[j] += SDD(dd1, dd2)
            # SDD_CCD[j] += SDD(ccd1, ccd2)
    return ([x / 30.0 for x in KSD_DD], [x / 30.0 for x in KSD_CCD])

def caculateSST(G, fracs):
    KSD_DD = [0] * len(fracs)
    KSD_CCD = [0] * len(fracs)
    ND_DD = [0] * len(fracs)
    ND_CCD = [0] * len(fracs)
    # SDD_DD = [0] * len(fracs)
    # SDD_CCD = [0] * len(fracs)
    reNodes = SST(G)
    for j in range(len(fracs)):
        tempNodes = []
        for node in reNodes:
            if len(tempNodes) >= fracs[j]*G.number_of_nodes():
                break
            tempNodes.append(node)
        sampleG = nx.subgraph(G, tempNodes)
        dd2 = DD(sampleG)
        ccd2 = CCD(sampleG)
        KSD_DD[j] = KSD(dd1, dd2)
        KSD_CCD[j] = KSD(ccd1, ccd2)
        ND_DD[j] = ND(dd1, dd2)
        ND_CCD[j] = ND(ccd1, ccd2)
        # SDD_DD[j] = SDD(dd1, dd2)
        # SDD_CCD[j] = SDD(ccd1, ccd2)
    return KSD_DD, KSD_CCD, ND_DD, ND_CCD



def drawP(G, distances):
    #在一个网络上，不同的比较指标随着p的变化趋势
    P = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    tempMysample = []
    for p in P:
        print p
        tempMysample.append(caculateMySample(G, 0.3, p))

    for i in range(6):
        plt.plot(P, [x[i] for x in tempMysample], label=distances[i])

    plt.title(G.name)
    plt.xlabel('P')
    plt.ylabel('Distance')
    plt.legend(loc=1)
    plt.show()

def drawF(G, distances):
    #在一个网络上，不同的比较指标随着p的变化趋势
    f = open(G.name + 'Results.txt', 'w')
    F = [x/100.0 for x in [0.2, 0.4, 0.6, 0.8, 1.0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]]
    tempMysample = caculateMySample(G, F, 1)
    # tempMysample1 = caculateMySample2(G, F, 1)
    # tempMysample2 = caculateMySample2(G, F, 0)
    # tempMysample3 = caculateMySample2(G, F, -1)
    # tempMysample4 = caculateMySample2(G, F, -1.5)
    # tempMysample5 = caculateMySample2(G, F, -2)
    tempRW = caculateRW(G, F)
    tempRWI = caculateRWI(G, F)
    # tempMysample2 = caculateMySample2(G, F, -1)
    # tempRWI2 = caculateRWI2(G, F)
    tempMHRW = caculateMHRW(G, F)
    tempMHRWI = caculateMHRWI(G, F)
    # tempMHRW2 = caculateMHRW2(G, F)
    tempFFS = caculateFFS(G, F)
    tempFFI = caculateFFI(G, F)
    tempBFS = caculateBFS(G, F)
    # tempRND = caculateRND(G, F)

    f.write('P')
    for i in F:
        f.write(' '+str(i))
    f.write('\n')

    fig, ax = plt.subplots(nrows=1, ncols=2)
    for i in range(2):
        f.write(distances[i]+'\n')

        f.write('Mysample')
        for j in tempMysample[i]:
            f.write(' '+str(j))
        f.write('\n')

        f.write('RW')
        for j in tempRW[i]:
            f.write(' ' + str(j))
        f.write('\n')

        f.write('RWI')
        for j in tempRWI[i]:
            f.write(' ' + str(j))
        f.write('\n')

        f.write('MHRW')
        for j in tempMHRW[i]:
            f.write(' ' + str(j))
        f.write('\n')

        f.write('MHRWI')
        for j in tempMHRWI[i]:
            f.write(' ' + str(j))
        f.write('\n')

        f.write('FFS')
        for j in tempFFS[i]:
            f.write(' ' + str(j))
        f.write('\n')

        f.write('FFI')
        for j in tempFFI[i]:
            f.write(' ' + str(j))
        f.write('\n')

        f.write('BFS')
        for j in tempBFS[i]:
            f.write(' ' + str(j))
        f.write('\n')

        plt.subplot(1, 2, i+1)
        plt.plot(F, tempMysample[i], label='Mysample', marker='o')
        # plt.plot(F, tempMysample1[i], label='MySample2_1')
        # plt.plot(F, tempMysample2[i], label='MySample2_-0')
        # plt.plot(F, tempRND[i], label='RND', marker='<')

        plt.plot(F, tempRW[i], label='RW', marker='s')
        plt.plot(F, tempRWI[i], label='RWI', marker='*')
        # plt.plot(F, tempMysample2[i], label='Mysample2', marker='^')
        # plt.plot(F, tempRWI2[i], label='RWI2', marker='s')
        plt.plot(F, tempMHRW[i], label='MHRW', marker='>')
        plt.plot(F, tempMHRWI[i], label='MHRWI', marker='^')
        plt.plot(F, tempFFS[i], label='FFS', marker='<')
        plt.plot(F, tempFFI[i], label='FFI', marker='v')
        plt.plot(F, tempBFS[i], label='BFS', marker='x')
        plt.title(G.name)
        plt.xlabel('F')
        plt.ylabel(distances[i])
        if i<2:
            plt.ylim((0, 1))
        plt.legend(loc=1)
    f.close()
    plt.show()






if __name__ == '__main__':
    # Email = constructG('Email')
    # powernet = constructG('powernet')

    # AstroPh = constructG('out.ca-AstroPh')
    # amazon = constructG('out.com-amazon')
    # douban = constructG('out.douban')
    # brightkite = constructG('out.loc-brightkite_edges')
    # gowalla = constructG('out.loc-gowalla_edges')
    # wordnet = constructG('out.wordnet-words')
    # livemocha = constructG('out.livemocha')
    flickrEdges = constructG('out.flickrEdges')

    # networks = [livemocha, AstroPh, amazon, douban, flickrEdges, brightkite, gowalla, wordnet]
    dd1 = DD(flickrEdges)
    ccd1 = CCD(flickrEdges)
    # # print dd1
    # # print ccd1
    # #
    distances = ['KSD_DD', 'KSD_CCD', 'ND_DD', 'ND_CCD']
    begin = time.clock()
    drawF(flickrEdges, distances)
    print time.clock() - begin










