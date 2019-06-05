import networkx as nx
import numpy as np
import math

def TK(temporalG):#时序Kshell
    W = {}
    tk = dict((k, 0) for k in temporalG[0].nodes())
    for G in temporalG:
        tempKshell = nx.core_number(G)
        print(tempKshell)
        for edge in G.edges():
            w = max(tempKshell[edge[0]], tempKshell[edge[1]])
            if edge in W.keys():
                W[edge] += w
            elif (edge[1], edge[0]) in W.keys():
                W[(edge[1], edge[0])] += w
            else:
                W[edge] = w
    for edge in W.keys():
        tk[edge[0]] += W[edge]
        tk[edge[1]] += W[edge]
    return sorted(tk.items(), key=lambda x: x[1], reverse=True)


def makeDataAll(G, beta, miu, name):
    Fect = {}
    for i in range(30):
        BA = G[i]
        if i == 0:
            for node in BA.nodes():
                Fect[node] = [node]

        Kshell = nx.core_number(BA)

        allNeighborsDegree = {}
        for node in BA.nodes():
            allNeighborsDegree[node] = 0
            for neighbor in BA.neighbors(node):
                allNeighborsDegree[node] += G.degree(neighbor)


        for node in BA.nodes():
            #degree
            Fect[node].append(float(BA.degree(node)))
            #kshell
            Fect[node].append(float(Kshell[node]))
            #neighborsDegree
            Fect[node].append(float(allNeighborsDegree[node]))
    for node in G[0].nodes():
        Fect[node].append(SIR(G, node, beta, miu))

    X = np.array([x for x in Fect.values()], dtype=float)
    np.save("data_"+name+"_D_"+str(beta)+"_"+str(miu)+".npy", X)

    return True

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

def Eigen(temporalG):
    N = len(temporalG[0].nodes())
    L = len(temporalG)

TSF = []
for i in range(30):
    BA = nx.read_edgelist('TemporalBA/BA' + str(i))
    TSF.append(BA)
# print(TSF[0].number_of_nodes(), TSF[0].number_of_edges())
# print(TSF[0].degree)
# print(nx.core_number(TSF[0]))
# print(TK(TSF))
print(TK(TSF))
print(TemporalDegreeDeviation(TSF))
