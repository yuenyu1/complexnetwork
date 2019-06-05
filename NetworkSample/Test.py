#coding=utf-8

import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

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
            temp += max(0.01, float(G.degree(neighbor)**u))
        r = random.uniform(0, 1)
        temp2 = 0
        for neighbor in G.neighbors(select):
            temp2 += max(0.01, float(G.degree(neighbor)**u))
            if temp2/temp >= r:
                if flag[neighbor] == 0:
                    infected.append(neighbor)
                    flag[neighbor] = 1
                break
        # print len(infected)
        if len(infected) >= frac*G.number_of_nodes():
            return infected
    return infected

def CCD(G):
    if G.number_of_nodes() == 0:
        return []
    re = [0]*101
    for i in nx.clustering(G).values():
        re[int(i*100)] += 1
    temp = [z/float(sum(re)) for z in re]
    # print temp
    return temp

def DD(G):
    if G.number_of_nodes() == 0:
        return []
    x = nx.degree_histogram(G)
    return [z/float(sum(x)) for z in x]

def KSD(d1, d2):
    d2 = d2 + (len(d1)-len(d2))*[0]
    maxD = 0
    for i in range(len(d1)):
        if abs(d1[i] - d2[i]) > maxD:
            maxD = abs(d1[i] - d2[i])
    return maxD

def caculateMySample(G, fracs, u, dd1, ccd1):
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

def draw1(networks, distances):
    # 在一个网络上，不同的比较指标随着p的变化趋势
    # f = open(G.name + 'Results.txt', 'w')
    F = [0.1]
    x = [m/10.0 for m in range(-20, 10, 2)]
    maker = ['o', 's', '*', '>', '^', '<', 'v', 'D', 'h', '+']
    KSD = []
    for j in range(len(networks)):
        tempKSD = [[], []]
        try:
            data = open(networks[j], 'r')
            for line in data.readlines():
                tempKSD[0].append(float(line.split(',')[0]))
                tempKSD[1].append(float(line.split(',')[1]))
        except:

            output = open(networks[j], 'w')
            G = constructG(networks[j])
            dd1 = DD(G)
            ccd1 = CCD(G)

            for k in x:
                tempKSD_DD, tempKSD_CCD = caculateMySample(G, F, k, dd1, ccd1)
                output.write(str(tempKSD_DD[0]) + ',' + str(tempKSD_CCD[0]) + '\n')
                tempKSD[0].append(tempKSD_DD[0])
                tempKSD[1].append(tempKSD_CCD[0])
            output.close()
        KSD.append(tempKSD)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        finalKSD = [0]*15
        for j in range(len(networks)):
            print KSD[j][i]
            for k in range(15):
                finalKSD[k] += KSD[j][i][k]
        print finalKSD
        plt.plot(x, [float(y)/10.0 for y in finalKSD], label=networks[j].replace('out.', ''), marker=maker[j])
        # plt.plot(F, tempMysample1[i], label='Mysample_-1.5', marker='s')
        # plt.plot(F, tempMysample2[i], label='Mysample_-1', marker='*')
        # plt.plot(F, tempMysample3[i], label='Mysample_-0.5', marker='>')
        # plt.plot(F, tempMysample4[i], label='Mysample_0', marker='^')
        # plt.plot(F, tempMysample5[i], label='Mysample_0.5', marker='<')
        # plt.plot(F, tempMysample6[i], label='Mysample_1', marker='v')
        plt.title('Ratio_0.1')
        plt.xlabel('x')
        plt.ylabel(distances[i])
        # if i < 2:
        #     plt.ylim((0, 1))
        plt.legend(loc=0, fontsize=10)
    plt.savefig('all1')

def SIR(G, infected, beta, miu):
    N = 1000
    re = 0
    while N > 0:
        inf = set()
        inf.add(infected)
        R = set()
        while len(inf) != 0:
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
        re += len(R)
        N -= 1
    return re/1000.0

# networks = ['Email', 'powernet', 'out.ca-AstroPh', 'out.com-amazon', 'out.douban', 'out.loc-brightkite_edges', 'out.loc-gowalla_edges', 'out.wordnet-words', 'out.livemocha', 'out.flickrEdges']
distances = ['KSD_DD', 'KSD_CCD']
networks = ['Email', 'powernet', 'out.ca-AstroPh', 'out.com-amazon', 'out.douban', 'out.loc-brightkite_edges', 'out.loc-gowalla_edges', 'out.wordnet-words', 'out.livemocha', 'out.flickrEdges']
draw1(networks, distances)
# heatConduction(Email, 0.2, 0)
