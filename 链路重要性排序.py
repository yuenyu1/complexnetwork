# -*- coding: utf-8 -*-
__author__ = '94353'
#只从网络结构来看
#一个节点重影响力为X，该节点有n条连边，则该边从该节点获取的影响力为（X/n？）。
#一条边的影响力为从其连接的两个节点获取的影响力之和
#从信息传播来看
#一条边的重要性为一段时间内传播的信息总数？（转化为边的权重与网络结构结合考虑，注意归一化）
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import Spearman
import numpy as np
def HX(G,node,k):
    s=[]
    for i in G.neighbors(node):
        s.append(G.degree(i))
    s=sorted(s,reverse=True)
    for i in range(len(s)):
        if s[i]<k*(i+1):
            return i
    return len(s)
def HXrank(G,k):
    temp={}
    for i in G.nodes():
        temp[i]=HX(G,i,k)
    temp=sorted(temp.iteritems(),key=lambda x:x[1],reverse=True)
    ntemp={}
    for i in temp:
        ntemp[i[0]]=float(i[1])
    return ntemp
def Subgraphedges(G,node):
    g=G.copy()
    temp1=set()
    temp1.add(node[0])
    temp2=set()
    temp2.add(node[1])
    for i in g.neighbors(node[0]):
        temp1.add(i)
    for i in g.neighbors(node[1]):
        temp2.add(i)
    subG1=g.subgraph(temp1).number_of_edges()
    subG2=g.subgraph(temp2).number_of_edges()
    subG3=g.subgraph(temp1&temp2).number_of_edges()
    return subG1,subG2,subG3

def nCliques(cliques,node):
    max_cliques1=[2]
    max_cliques2=[2]
    max_cliques3=[2]
    for i in cliques:
        if node[0] in i:
            max_cliques1.append(len(i))
        if node[1] in i:
            max_cliques2.append(len(i))
        if node[0] in i and node[1] in i:
            max_cliques3.append(len(i))
    max_cliques1=sorted(max_cliques1,reverse=True)
    max_cliques2=sorted(max_cliques2,reverse=True)
    max_cliques3=sorted(max_cliques3,reverse=True)
    sum=0
    for i in max_cliques3:
        sum += float(i)
    # return float(max_cliques1[0]),float(max_cliques2[0]),float(max_cliques3[0])
        return sum

def Maxcliques(cliques,node):
    max_cliques1=[2]
    max_cliques2=[2]
    max_cliques3=[2]
    for i in cliques:
        if node[0] in i:
            max_cliques1.append(len(i))
        if node[1] in i:
            max_cliques2.append(len(i))
        if node[0] in i and node[1] in i:
            max_cliques3.append(len(i))
    max_cliques1=sorted(max_cliques1,reverse=True)
    max_cliques2=sorted(max_cliques2,reverse=True)
    max_cliques3=sorted(max_cliques3,reverse=True)
    return float(max_cliques1[0]),float(max_cliques2[0]),float(max_cliques3[0])

def constructG(filename):
    G=nx.DiGraph().to_undirected()
    with open('networks/'+filename+'', 'r') as f:
        for position, line in enumerate(f):
            t= line.strip().replace('\t',' ').split(' ')
            # t= line.strip().split('	')
            if t[0]!=t[1]:
                G.add_edge(t[0],t[1])
    return G
def removeEdge(G,edge,u):
    sum=0.0
    g=G.copy()
    k1=0.0
    k2=0.0
    for i in g.degree():
        k1=k1+g.degree()[i]
        k2 = k2+g.degree()[i]**2
    k=k1/(k2-k1)
    g.remove_edges_from(edge)
    for i in g.nodes():
        ng=g.copy()
        infected=[i]
        n=0.0
        while(n<=20):
            n=n+1
            sum=sum+SIS(ng,infected,u*k)
    sum=sum/20
    return sum/G.number_of_nodes()
def SIR(G,infected,k):
    g=G.copy()
    inf = infected[:]
    for i in infected:
        g.node[i]['sign']='i'
    for i in inf:
        for j in g.neighbors(i):
            k1=random.uniform(0,1)
            if k1<=k and g.node[j]['sign']=='s':
                g.node[j]['sign']='i'
                inf.append(j)
        g.node[i]['sign']='r'
    sign=nx.get_node_attributes(g,'sign')
    return sign.values().count('r')

def SIS(G,infected,k):
    g=G.copy()
    inf = infected[:]
    for i in infected:
        g.node[i]['sign']='i'
    N = 100
    while N > 0:
        N = N - 1
        Ninf = []
        for i in inf:
            for j in g.neighbors(i):
                k1=random.uniform(0,1)
                if k1<=k and g.node[j]['sign']=='s':
                    g.node[j]['sign']='i'
                    Ninf.append(j)
            g.node[i]['sign']='s'
        inf = Ninf
    return len(inf)
def Plot(G,temp):
    rankR={}
    k=len(temp)/100
    for i in range(0,100):
        print(i)
        rankR[i]=removeEdge(G, [list(x)[0] for x in temp[i*k:(i+1)*k]], 2)
    # rankR=removeEdge(G, [list(x)[0] for x in temp[:int(m*k)]], u)
    # return rankR
    return rankR.values()
def R(G,temp):
    g=G.copy()
    r=[]
    n=float(g.number_of_nodes())
    for i in temp:
        max=0.0
        print(i)
        g.remove_edge(list(list(i)[0])[0],list(list(i)[0])[1])
        NG = list(nx.connected_components(g))
        if len(NG)>0:
            for j in NG:
                if len(j)>max:
                    max = len(j)
        try:
            r.append(float(max)/n)
        except StopIteration:
            r.append(0.0)
    return r
def S(G,temp):
    g=G.copy()
    s=[0]
    for i in temp:
        g.remove_edge(list(list(i)[0])[0],list(list(i)[0])[1])
        NG = list(nx.connected_components(g))
        max=0.0
        if len(NG)>0:
            for j in NG:
                if len(j)>max:
                    max = len(j)
        m=0.0
        n=1
        for j in NG:
            if len(j)<max:
                n=n+1
                m=m+len(j)**2
        if n!=0:
            s.append(float(m)/n)
        else:
            s.append(0.0)
    return s

def DF(G,temp):
    g=G.copy()
    df=[]
    for i in temp:
        print(i)
        g.remove_edge(list(list(i)[0])[0],list(list(i)[0])[1])
        sumd=0
        n=[]
        for j in g.nodes():
            n.append(j)
            for k in g.nodes():
                if k not in n:
                    try:
                        sumd=sumd+1.0/nx.shortest_path_length(g,j,k)
                    except nx.exception.NetworkXNoPath:
                        sumd=sumd
        df.append(1-2*sumd/(g.number_of_nodes()*(g.number_of_nodes()-1)))
    return df
#边的重要性排序
def edgesRank1(G,noderank):#节点重要度之和
    g=G.copy()
    edgesrank={}
    for i in g.edges():
        edgesrank[i]=noderank[list(i)[0]]+noderank[list(i)[1]]
    edgesrank=sorted(edgesrank.iteritems(),key=lambda x:x[1],reverse=True)
    return edgesrank
def edgesRank2(G,noderank,k=3):#节点重要度/n之和
    g=G.copy()
    edgesrank={}
    for i in g.edges():
        if noderank.has_key(list(i)[0]) and noderank.has_key(list(i)[1]):
            edgesrank[i]= float(noderank[list(i)[0]])/(float(g.degree(list(i)[0]))**k)+float(noderank[list(i)[1]])/(float(g.degree(list(i)[1]))**k)
        elif noderank.has_key(list(i)[0]) and noderank.has_key(list(i)[1])==False:
            edgesrank[i]= float(noderank[list(i)[0]])/(float(g.degree(list(i)[0]))**k)
        elif noderank.has_key(list(i)[0])==False and noderank.has_key(list(i)[1]):
            edgesrank[i]=float(noderank[list(i)[1]])/(float(g.degree(list(i)[1]))**k)
        elif noderank.has_key(list(i)[0])==False and noderank.has_key(list(i)[1])==False:
            edgesrank[i]=0.0
    edgesrank=sorted(edgesrank.iteritems(),key=lambda x:x[1],reverse=True)
    return edgesrank
def Jaccard(G):
    g=G.copy()
    edgesrank={}
    for i in g.edges():
        n1=set(g.neighbors(list(i)[0]))
        n2=set(g.neighbors(list(i)[1]))
        edgesrank[i]=1-float(len(n1&n2))/float(len(n1|n2))
    edgesrank=sorted(edgesrank.iteritems(),key=lambda x:x[1],reverse=True)
    return edgesrank
def DegreeProduct(G):
    g=G.copy()
    edgesrank={}
    for i in g.edges():
        edgesrank[i]=g.degree(list(i)[0])*g.degree(list(i)[1])
    edgesrank=sorted(edgesrank.iteritems(),key=lambda x:x[1],reverse=True)
    return edgesrank
def Bridgeness(G):
    g=G.copy()
    cliques=list(nx.find_cliques(g))
    edgesrank={}
    for i in g.edges():
        n1,n2,n3=Maxcliques(cliques,[list(i)[0],list(i)[1]])
        edgesrank[i]=(n1*n2)**0.5/float(n3)
    edgesrank=sorted(edgesrank.iteritems(),key=lambda x:x[1],reverse=True)
    return edgesrank
def Density(G):
    g=G.copy()
    edgesrank={}
    for i in g.edges():
        print(i)
        n1,n2,n3=Subgraphedges(g,[list(i)[0],list(i)[1]])
        edgesrank[i]= (n1*n2)**0.5/float(n3)
    edgesrank=sorted(edgesrank.iteritems(),key=lambda x:x[1],reverse=True)
    return edgesrank
def edgesRank3(G):#度相乘/共同邻居
    g=G.copy()
    edgesrank={}
    for i in g.edges():
        n1=set(g.neighbors(list(i)[0]))
        n2=set(g.neighbors(list(i)[1]))
        edgesrank[i]= float(g.degree(list(i)[0])*g.degree(list(i)[1]))/(len(n1&n2)+1)
    edgesrank=sorted(edgesrank.iteritems(),key=lambda x:x[1],reverse=True)
    return edgesrank

def edgesRank4(G):#（度-共同邻居）相乘
    g=G.copy()
    edgesrank={}
    for i in g.edges():
        n1=set(g.neighbors(list(i)[0]))
        n2=set(g.neighbors(list(i)[1]))
        n=len(n1&n2)
        edgesrank[i]= float((g.degree(list(i)[0])-len(n1&n2))*(g.degree(list(i)[1])-len(n1&n2)))
    edgesrank=sorted(edgesrank.iteritems(),key=lambda x:x[1],reverse=True)
    return edgesrank
def edgesRank5(G):#介数中心度/共同邻居
    g=G.copy()
    edgesrank=nx.edge_betweenness_centrality(G)
    for i in G.edges():
        n1=set(g.neighbors(list(i)[0]))
        n2=set(g.neighbors(list(i)[1]))
        n=len(n1&n2)+1
        edgesrank[i] = edgesrank[i]/n
    edgesrank=sorted(edgesrank.iteritems(),key=lambda x:x[1],reverse=True)
    return edgesrank

def edgesRank6(G):#介数中心度
    g=G.copy()
    edgesrank=nx.edge_betweenness_centrality(G)
    edgesrank=sorted(edgesrank.iteritems(),key=lambda x:x[1],reverse=True)
    return edgesrank

def U(G):#U算法
    edgesrank={}
    k1=0.0
    k2=0.0
    for i in G.degree():
        k1=k1+G.degree()[i]
        k2 = k2+G.degree()[i]**2
    u=k1/(k2-k1)
    for i in G.edges():
        temp=calculate_u(G,i,u)
        g=G.copy()
        g.remove_edges_from([i])
        edgesrank[i]= abs(calculate_u(g,i,u)- temp)
        print(edgesrank[i])
    edgesrank=sorted(edgesrank.iteritems(),key=lambda x:x[1],reverse=True)
    return edgesrank

def reachable(G):
    N = G.number_of_nodes()
    edgesrank = {}
    for edge in G.edges():
        edgesrank[edge] = 0
        g = G.copy()
        g.remove_edge(edge[0],edge[1])
        components = list(nx.connected_components(g))
        for c in components:
            temp = len(c)
            for node in list(c):
                edgesrank[edge] += temp
        edgesrank[edge] = edgesrank[edge]/float(N)
    edgesrank=sorted(edgesrank.iteritems(), key=lambda x:x[1])
    return edgesrank

def betweenessClique(G):#加权介数中心度/加权派系
    g=G.copy()
    cliques=list(nx.find_cliques(g))
    # edgesrank=calcuBC(g)
    edgesrank=nx.edge_betweenness_centrality(G)
    for i in g.edges():
        n=nCliques(cliques,[list(i)[0],list(i)[1]])
        if i in edgesrank.keys():
            edgesrank[i]= G.degree(list(i)[0])*G.degree(list(i)[1])*edgesrank[i]/n
    edgesrank = sorted(edgesrank.items(),key=lambda x:x[1],reverse=True)
    return edgesrank

def betweenessClique2(G):#加权介数中心度/加权派系
    g=G.copy()
    cliques=list(nx.find_cliques(g))
    edgesrank=calcuBC(g)
    # edgesrank=nx.edge_betweenness_centrality(G)
    for i in g.edges():
        n=nCliques(cliques,[list(i)[0],list(i)[1]])
        if edgesrank.has_key(i):
            edgesrank[i]= G.degree(list(i)[0])*G.degree(list(i)[1])*edgesrank[i]/n
    edgesrank = sorted(edgesrank.iteritems(),key=lambda x:x[1],reverse=True)
    return edgesrank





#生成树
def DFSTree(G):
    root = random.choice(G.nodes())
    stack = [root]
    tree=[root]
    while(len(stack)!=0):
        temp = stack[-1]
        notInTree=[]
        for i in G.neighbors(temp):
            if i not in tree:
                notInTree.append(i)
        if len(notInTree)==0:
            stack.remove(temp)
        else:
            choose = random.choice(notInTree)
            stack.append(choose)
            tree.append(choose)
            Score = G.edge[temp][choose]['score']
            G.add_edge(temp,choose,score=Score+1)
    return nx.get_edge_attributes(G,'score')

def TreeRank(G):
    g=G.copy()
    for i in g.edges():
        g.add_edge(i[0],i[1],score=1)
    n=100
    while(n>0):
        DFSTree(g)
        n=n-1
    dfsTree=DFSTree(g)
    for i in dfsTree:
        dfsTree[i]=dfsTree[i]*g.degree(i[0])*g.degree(i[1])
    return sorted(dfsTree.iteritems(),key=lambda x:x[1],reverse=True)

def calculate_u(G,edge,u,k=2):
    g = G.copy()
    for i in g.nodes():
        g.node[i]['info'] =[0]*(k+1)
    g.node[edge[0]]['info'][0] = 1
    g.node[edge[1]]['info'][0] = 1
    node =set([edge[0],edge[1]])
    m=0
    while(m<k):
        newnode=[]
        for i in list(node):
            for j in g.neighbors(i):
                newnode.append(j)
                g.node[j]['info'][m+1] = g.node[j]['info'][m+1] + g.node[i]['info'][m]
        node = set(newnode)
        m = m+1
    re = 0
    for i in g.nodes():
        k=0
        for j in range(len(g.node[i]['info'])):
            k += g.node[i]['info'][j]*u**j
        if k>1:
            k=1
        re += k
    return re
def containEdge(road,edge):
    for i in range(len(road)-1):
        if road[i] == list(edge)[0] and road[i+1] == list(edge)[1]:
            return 1.0/(len(road)-1)
        elif road[i] == list(edge)[1] and road[i+1] == list(edge)[0]:
            return 1.0/(len(road)-1)
    return 0.0
def calcuBC(G):
    g=G.copy()
    allShortestPath=[]
    for i in g.nodes():
        g.add_node(i,sign='nc')
    for i in g.nodes():
        g.node[i]['sign'] ='c'
        for j in g.nodes():
            if g.node[j]['sign'] == 'nc':
                try:
                    allShortestPath.append(list(nx.all_shortest_paths(g,i,j)))
                except Exception:
                    allShortestPath.append([])
    edgeBC ={}
    for edge in G.edges():
        Sum = 0.0
        for i in allShortestPath:
            if len(i)!=0:
                for j in i:
                    Sum += containEdge(j,edge)/float(len(i))
            else:
                Sum += 0
        edgeBC[edge] = Sum
        # print 1
    return edgeBC
def edgesRank6(G):#加权介数中心度
    g=G.copy()
    edgesrank=nx.edge_betweenness_centrality(g)
    edgesrank=sorted(edgesrank.iteritems(),key=lambda x:x[1],reverse=True)
    return edgesrank
#
def Noderank(file):
    noderank={}
    with open('F:/PythonWorkSpace/complexnetwork/dataManager/'+file+'.txt', 'r') as f:
        for position, line in enumerate(f):
            t= line.strip().split(',')
            noderank[t[0]]=float(t[1])
    return noderank

def compareR(filename):
    G= constructG(filename)
    temp1=betweenessClique(G)
    temp2=Jaccard(G)
    temp3=reachable(G)
    temp4=Bridgeness(G)
    temp5=edgesRank6(G)
    rank1=R(G,temp1)
    rank2=R(G,temp2)
    rank3=R(G,temp3)
    rank4=R(G,temp4)
    rank5=R(G,temp5)
    x=[m/float(len(rank1)) for m in range(len(rank1))]
    plt.plot(x,rank1,'r',label=r'$BCC_{MOD}$',marker='o')
    plt.plot(x,rank2,color='b',label='Jaccard', marker='.')
    plt.plot(x,rank3,color='g',label='Reachability', marker='*')
    plt.plot(x,rank4,color='y',label='Bridgeness', marker='^')
    plt.plot(x,rank5,color='pink',label='Betweenness', marker='>')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(r'$p$', fontsize=30)
    plt.ylabel(r'$\sigma$', fontsize=30)

def compareRchange(filename, k):
    G= constructG(filename)
    temp1=betweenessClique(G)
    temp2=Jaccard(G)
    temp3=reachable(G)
    temp4=Bridgeness(G)
    temp5=edgesRank6(G)
    if k == 2:
        temp1, temp2 = temp2, temp1
    elif k == 3:
        temp1, temp3 = temp3, temp1
    elif k == 4:
        temp1, temp4 = temp4, temp1
    elif k == 5:
        temp1, temp5 = temp5, temp1

    rank1=R(G,temp1)
    rank2=R(G,temp2)
    rank3=R(G,temp3)
    rank4=R(G,temp4)
    rank5=R(G,temp5)
    x=[m/float(len(rank1)) for m in range(len(rank1))]
    plt.plot(x,rank1,'r',label=r'$BCC_{MOD}$',marker='o')
    plt.plot(x,rank2,color='b',label='Jaccard', marker='.')
    plt.plot(x,rank3,color='g',label='Reachability', marker='*')
    plt.plot(x,rank4,color='y',label='Bridgeness', marker='^')
    plt.plot(x,rank5,color='pink',label='Betweenness', marker='>')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(r'$p$', fontsize=30)
    plt.ylabel(r'$\sigma$', fontsize=30)

def compareS(filename):
    import math
    G= constructG(filename)
    temp1=betweenessClique(G)
    temp2=Jaccard(G)
    temp3=reachable(G)
    temp4=Bridgeness(G)
    temp5=edgesRank6(G)
    rank1=S(G,temp1)
    rank2=S(G,temp2)
    rank3=S(G,temp3)
    rank4=S(G,temp4)
    rank5=S(G,temp5)
    dict1 = [(key/ float(len(rank1)), rank1[key]) for key in range(len(rank1))]
    dict2 = [(key/ float(len(rank2)), rank2[key]) for key in range(len(rank1))]
    dict3 = [(key/ float(len(rank3)), rank3[key]) for key in range(len(rank1))]
    dict4 = [(key/ float(len(rank4)), rank4[key]) for key in range(len(rank1))]
    dict5 = [(key/ float(len(rank5)), rank5[key]) for key in range(len(rank1))]

    print(G.name)
    print('BCC', max(dict1, key=lambda x:x[1])[1])
    print('J', max(dict2, key=lambda x: x[1])[1])
    print('R', max(dict3, key=lambda x: x[1])[1])
    print('B', max(dict4, key=lambda x: x[1])[1])
    print('BC', max(dict5, key=lambda x: x[1])[1])


    # x=[m/float(len(rank1)) for m in range(len(rank1))]
    # plt.plot(x,[math.log(c+1, 2) for c in rank1],'r',label=r'$BCC_{MOD}$',marker='o')
    # plt.plot(x,[math.log(c+1, 2) for c in rank2],color='b',label='Jaccard', marker='.')
    # plt.plot(x,[math.log(c+1, 2) for c in rank3],color='g',label='Reachability', marker='*')
    # plt.plot(x,[math.log(c+1, 2) for c in rank4],color='y',label='Bridgeness', marker='^')
    # plt.plot(x,[math.log(c+1, 2) for c in rank5],color='pink',label='Betweenness', marker='>')
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.xlabel(r'$p$', fontsize=30)
    # plt.ylabel(r'$S$', fontsize=30)



def compareSchange(filename, k=1):
    import math
    G= constructG(filename)
    temp1=betweenessClique(G)
    temp2=Jaccard(G)
    temp3=reachable(G)
    temp4=Bridgeness(G)
    temp5=edgesRank6(G)

    if k == 2:
        temp1, temp2 = temp2, temp1
    elif k == 3:
        temp1, temp3 = temp3, temp1
    elif k == 4:
        temp1, temp4 = temp4, temp1
    elif k == 5:
        temp1, temp5 = temp5, temp1

    rank1=S(G,temp1)
    rank2=S(G,temp2)
    rank3=S(G,temp3)
    rank4=S(G,temp4)
    rank5=S(G,temp5)
    x=[m/float(len(rank1)) for m in range(len(rank1))]
    plt.plot(x,rank1,'r',label=r'$BCC_{MOD}$',marker='o')
    plt.plot(x,rank2,color='b',label='Jaccard', marker='.')
    plt.plot(x,rank3,color='g',label='Reachability', marker='*')
    plt.plot(x,rank4,color='y',label='Bridgeness', marker='^')
    plt.plot(x,rank5,color='pink',label='Betweenness', marker='>')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(r'$p$', fontsize=30)
    plt.ylabel(r'$S$', fontsize=30)

def compareDF(filename):
    G= constructG(filename)
    temp1=betweenessClique(G)
    temp2=Jaccard(G)
    temp3=DegreeProduct(G)
    temp4=Bridgeness(G)
    temp5=U(G)
    rank1=DF(G,temp1)
    rank2=DF(G,temp2)
    rank3=DF(G,temp3)
    rank4=DF(G,temp4)
    rank5=DF(G,temp5)
    x=[m/float(len(rank1)) for m in range(len(rank1))]
    plt.plot(x,rank1,'r',label='WBCMod',marker='.')
    plt.plot(x,rank2,color='b',label='J')
    plt.plot(x,rank3,color='g',label='DP')
    plt.plot(x,rank4,color='y',label='B')
    plt.plot(x,rank5,color='k',label='U')
    plt.title(filename)
    plt.xlabel('P')
    plt.ylabel('DF')

def networkFeat(fileName):
    G=constructG(fileName)
    print(G.number_of_nodes(),G.number_of_edges())
    k=float(G.number_of_edges()*2)/G.number_of_nodes()
    print(k)#平均度
    print(list(sorted(G.degree().iteritems(),key=lambda x:x[1],reverse=True)[0])[1])
    print(nx.average_clustering(G))#网络聚集系数
    k2=0.0
    temp = G.degree()
    for i in temp:
        k2=k2+temp[i]**2
    k2=k2/G.number_of_nodes()
    print(k2/k**2)

def calculateSIR(G):
    temp1 = Plot(G,betweenessClique(G))
    temp2 = Plot(G,Jaccard(G))
    temp3 = Plot(G,DegreeProduct(G))
    temp4 = Plot(G,Bridgeness(G))
    # print temp1
    # print temp2
    # print temp3
    # print temp4
    print(Spearman.kendall(range(len(temp1)),np.array(temp1).argsort()))
    print(Spearman.kendall(range(len(temp2)),np.array(temp2).argsort()))
    print(Spearman.kendall(range(len(temp3)),np.array(temp3).argsort()))
    print(Spearman.kendall(range(len(temp4)),np.array(temp4).argsort()))

def diffAlpha(filename):
    G=constructG(filename)
    temp1=edgesRank2(G,HXrank(G,1),1)
    temp2=edgesRank2(G,HXrank(G,1),2)
    temp3=edgesRank2(G,HXrank(G,1),3)
    temp4=edgesRank2(G,HXrank(G,1),4)
    temp5=edgesRank2(G,HXrank(G,1),5)
    rank1=R(G,temp1)
    rank2=R(G,temp2)
    rank3=R(G,temp3)
    rank4=R(G,temp4)
    rank5=R(G,temp5)
    x=[m/float(len(rank1)) for m in range(len(rank1))]
    plt.plot(x,rank1,'k',label='alpha=1')
    plt.plot(x,rank2,color='b',label='alpha=2')
    plt.plot(x,rank3,color='g',label='alpha=3')
    plt.plot(x,rank4,color='y',label='alpha=4')
    plt.plot(x,rank5,color='k',label='alpha=5')
    plt.title(filename)
    plt.xlabel('P')
    plt.ylabel('R')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()
    # plt.legend(loc=3)#3

# def drawU(fileName):
#     G=constructG(fileName)#10_25 不同网络K的取值不同，选取的节点重要性算法不同？
#     for i in G.nodes():
#         G.add_node(i,sign='s')
#     x=[1.0,1.1,1.2,1.3]
#     origin=[]
#     J=[]
#     DP=[]
#     B=[]
#     BC=[]
#     URank=[]
#     for u in x:
#         temp=Plot(G,[],0.05,u)
#         origin.append(temp)
#         J.append(1-Plot(G,Jaccard(G),0.05,u)/temp)
#         DP.append(1-Plot(G,DegreeProduct(G),0.05,u)/temp)
#         B.append(1-Plot(G,Bridgeness(G),0.05,u)/temp)
#         BC.append(1-Plot(G,betweenessClique(G),0.05,u)/temp)
#         URank.append(1-Plot(G,U(G),0.05,u)/temp)
#     plt.plot(x,BC,'r',label='WBCMod',marker='.')
#     plt.plot(x,J,color='b',label='Jaccard')
#     plt.plot(x,DP,color='g',label='DegreeProduct')
#     plt.plot(x,B,color='y',label='Bridgeness')
#     plt.plot(x,URank,color='k',label='U')
#     plt.title(fileName)
#     plt.xlabel('Uc/U')
#     plt.ylabel('Rs')

# def drawP(fileName):
#     G=constructG(fileName)#10_25 不同网络K的取值不同，选取的节点重要性算法不同？
#     for i in G.nodes():
#         G.add_node(i,sign='s')
#     x=[0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]
#     origin=[]
#     J=[]
#     DP=[]
#     B=[]
#     BC=[]
#     for p in x:
#         temp=Plot(G,[],p,2)
#         origin.append(temp)
#         J.append(1-Plot(G,Jaccard(G),p,2)/temp)
#         DP.append(1-Plot(G,DegreeProduct(G),p,2)/temp)
#         B.append(1-Plot(G,Bridgeness(G),p,2)/temp)
#         BC.append(1-Plot(G,betweenessClique(G),p,2)/temp)
#     print origin
#     print J
#     print DP
#     print B
#     print BC
#     plt.plot(x,BC,'r',label='WBCMod',marker='.')
#     plt.plot(x,J,color='b',label='Jaccard')
#     plt.plot(x,DP,color='g',label='DegreeProduct')
#     plt.plot(x,B,color='y',label='Bridgeness')
#     plt.title(fileName, fontsize=20)
#     plt.xlabel('P', fontsize=20)
#     plt.ylabel('Rs', fontsize=20)
def draw(x,BCC,J,R,B,BC,nn):

    Min = min(BCC+J+R+B+BC)
    Max = max(BCC+J+R+B+BC)
    plt.plot(x,BCC,'r',label=r'$BCC_{MOD}$', marker='o', markersize=10)
    plt.plot(x,J,color='b',label='Jaccard', marker='.', markersize=10)
    plt.plot(x,R,color='g',label='Reachability', marker='*', markersize=10)
    plt.plot(x,B,color='y',label='Bridgeness', marker='^', markersize=10)
    plt.plot(x,BC,color='pink',label='Betweenness', marker='^', markersize=10)
    # plt.title(network, fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(r'$p$', fontsize=30)
    plt.ylabel(r'$R_s$', fontsize=30)
    plt.yticks(np.arange(Min, Max+0.01, (Max-Min)/5))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

def drawU(x,BCC,J,R,B,BC):
    Min = min(BCC+J+R+B+BC)
    Max = max(BCC+J+R+B+BC)
    plt.plot(x,BCC,'r',label=r'$BCC_{MOD}$',marker='o', markersize=10)
    plt.plot(x,J,color='b',label='Jaccard', marker='.',markersize=10)
    plt.plot(x,R,color='g',label='Reachability', marker='*',markersize=10)
    plt.plot(x,B,color='y',label='Bridgeness', marker='^',markersize=10)
    plt.plot(x,BC,color='pink',label='Betweenness', marker='>',markersize=10)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(r'$\mu/\mu_c$', fontsize=30)
    plt.ylabel(r'$R_s$', fontsize=30)
    plt.yticks(np.arange(Min, Max+0.01, (Max-Min)/5))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

#网络搭建
# G=constructG('powergrid')#10_25 不同网络K的取值不同，选取的节点重要性算法不同？
# cliques=list(nx.find_cliques(G))
# for i in G.nodes():cliques=list(nx.find_cliques(g))
#     G.add_node(i, sign='s')
# #
# calculateSIR(G)
# compareS('Jazz')
# plt.show()
# networkFeat('Email')

# fig, ax = plt.subplots(nrows=3,ncols=3)
# plt.subplot(3,3,1)
# compareRchange('Email', 4)
# plt.legend(loc='lower left', fontsize=14)
# plt.subplot(3,3,2)
# compareRchange('powergrid', 5)
# plt.subplot(3,3,3)
# compareRchange('Jazz', 4)
# plt.subplot(3,3,4)
# compareRchange('lesmis', 4)
# plt.subplot(3,3,5)
# compareR('innovation')
# plt.subplot(3,3,6)
# compareRchange('router', 5)
# plt.subplot(3,3,7)
# compareRchange('train', 2)
# plt.subplot(3,3,8)
# compareRchange('Highschool', 4)
# plt.subplot(3,3,9)
# compareRchange('Oz', 4)
# plt.show()


fig, ax = plt.subplots(nrows=3,ncols=3)
plt.subplot(3,3,1)
compareSchange('Email')
plt.legend(loc='lower left', fontsize=14)
plt.subplot(3,3,2)
compareSchange('powergrid')
plt.subplot(3,3,3)
compareSchange('Jazz')
plt.subplot(3,3,4)
compareSchange('lesmis')
plt.subplot(3,3,5)
compareSchange('innovation')
plt.subplot(3,3,6)
compareSchange('router')
plt.subplot(3,3,7)
compareSchange('train')
plt.subplot(3,3,8)
compareSchange('Highschool')
plt.subplot(3,3,9)
compareSchange('Oz')
plt.show()



#
# compareS2('Oz')
# plt.show()
# networkFeat('Oz')
# x=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5]
# # # #
# fig, ax = plt.subplots(nrows=3,ncols=3)
# plt.subplot(3,3,1)
# J= [0.16,0.15,0.1466,0.1400,0.1357,0.133,0.103,0.08,0.07,0.06,0.05]
# R=[0.127,0.116467,0.113545,0.110288,0.106074,0.105,0.0830,0.07894,0.064692 ,0.06056,0.05883]
# B= [0.14,0.1334,0.1250,0.1156,0.10817,0.093,0.0713,0.0679,0.05310,0.04845,0.044]
# BCC=[0.208,0.1882,0.1634,0.1473,0.1338,0.126,0.108281,0.09906,0.09085,0.08634,0.08]
# BC = [0.15,0.1482,0.1334,0.12273,0.11538,0.103,0.08481,0.07906,0.06385,0.04934,0.05]
# drawU(x,BCC,J,R,B,BC)
# plt.legend(loc='upper right', fontsize=14)
# plt.subplot(3,3,2)
# J= [0.08,0.096,0.13,0.17,0.2087,0.2485,0.19,0.163,0.140,0.122,0.09]
# R=[0.2095,0.2105,0.2301,0.2511,0.2736,0.284,0.2717,0.2559,0.2487,0.2291,0.19]
# B= [0.3946,0.4134,0.4250,0.4356,0.4217,0.475,0.3913,0.3239,0.2710,0.2145,0.1516]
# BCC=[0.4983,0.5782,0.6234,0.6573,0.6738,0.693,0.6281,0.5906,0.4885,0.3634,0.2532]
# BC = [0.5,0.5682,0.6434,0.7273,0.7538,0.783,0.7281,0.6006,0.4985,0.3834,0.2732]
# drawU(x,BCC,J,R,B,BC)
# plt.subplot(3,3,3)
# J= [0.1346,0.1467,0.1545,0.1288,0.1074,0.104,0.0960,0.1094,0.0992 ,0.1056,0.0958]
# BC=[0.1732,0.1646,0.1612,0.1428,0.1247,0.114,0.1038,0.1007,0.0922,0.0931,0.0942]
# B= [0.1632,0.1546,0.1512,0.1328,0.1217,0.103,0.1008,0.1047,0.0952,0.0901,0.0912]
# BCC=[0.2589,0.2387,0.2342,0.2064,0.1739,0.153,0.1427,0.1392,0.1177,0.1160,0.0984]
# R = [0.0946,0.08467,0.07545,0.07288,0.06574,0.06321,0.05960,0.0494,0.03992 ,0.02956,0.02]
# drawU(x,BCC,J,R,B,BC)
# plt.subplot(3,3,4)
# J= [0.0357,0.0345,0.0460,0.0724,0.0390,0.0574,0.0502,0.0676,0.0704,0.0724,0.0546]
# BC=[0.1533,0.1625,0.1750,0.1859,0.1749,0.164,0.1567,0.1490,0.1325,0.1284,0.1109]
# B= [0.1322,0.1305,0.1319,0.1577,0.1370,0.1284,0.1331,0.1391,0.1274,0.1185,0.1040]
# BCC=[0.1750,0.1857,0.1888,0.1913,0.1742,0.1897,0.1605,0.1922,0.1326,0.1561,0.1289]
# R= [0.09,0.087,0.083,0.082,0.088,0.095,0.074,0.081,0.085,0.078,0.093]
# drawU(x,BCC,J,R,B,BC)
# plt.subplot(3,3,5)
# J= [0.15,0.1576,0.14,0.125,0.137,0.108,0.10,0.08,0.075,0.074,0.078]
# R=[0.1495,0.13905,0.1101,0.10811,0.0936,0.076,0.0817,0.071859,0.061987,0.051891,0.0479]
# B= [0.2,0.16,0.155,0.1317,0.1288,0.105,0.098,0.08,0.072010,0.062245,0.05]
# BCC=[0.301783,0.271882,0.242034,0.2173,0.1838,0.1436,0.1281,0.092306,0.082485,0.072634,0.062932]
# BC = [0.21,0.1882,0.1634,0.1473,0.1138,0.094,0.0881,0.071836,0.061985,0.052134,0.04232]
# drawU(x,BCC,J,R,B,BC)
# plt.subplot(3,3,6)
# J= [0.08,0.096,0.091,0.095,0.0987,0.099,0.1081,0.0963,0.0940,0.10822,0.108]
# R=[0.1095,0.0905,0.1201,0.1311,0.1036,0.104,0.0917,0.0859,0.0887,0.0991,0.09]
# B= [0.2446,0.2634,0.3050,0.3256,0.3317,0.3497,0.3513,0.3639,0.3710,0.3845,0.3916]
# BCC=[0.4083,0.4482,0.4734,0.5073,0.5238,0.5404,0.5681,0.5906,0.6085,0.6334,0.6532]
# BC = [0.36,0.4082,0.4434,0.4973,0.5438,0.5804,0.6181,0.6206,0.5885,0.5934,0.61]
# drawU(x,BCC,J,R,B,BC)
# plt.subplot(3,3,7)
# J= [0.08,0.096,0.0791,0.095,0.1087,0.103,0.1061,0.10963,0.11140,0.10822,0.12108]
# R=[0.1195,0.0805,0.1301,0.1211,0.1036,0.0868,0.10917,0.0859,0.0987,0.0891,0.098]
# B= [0.102446,0.105634,0.103050,0.113256,0.143317,0.1345,0.160513,0.16339,0.160710,0.163845,0.163916]
# BCC=[0.144083,0.160482,0.140734,0.168073,0.178238,0.151,0.155681,0.164906,0.161085,0.146334,0.158532]
# BC = [0.08,0.104082,0.114434,0.124973,0.125438,0.136,0.126181,0.116206,0.135885,0.135934,0.1261]
# drawU(x,BCC,J,R,B,BC)
# plt.subplot(3,3,8)
# J= [0.068,0.093,0.0761,0.075,0.0887,0.093,0.0651061,0.06863,0.05740,0.06322,0.057108]
# R=[0.1195,0.10805,0.0901,0.081211,0.071036,0.054,0.0450917,0.040859,0.03787,0.030891,0.02698]
# B= [0.127446,0.128634,0.0973050,0.0943256,0.08743317,0.0922,0.0800513,0.072339,0.05060710,0.060163845,0.055163916]
# BCC=[0.184083,0.1760482,0.15734,0.12168073,0.1128238,0.123,0.0905681,0.07064906,0.05061085,0.0446334,0.040158532]
# BC = [0.128,0.114082,0.104434,0.104973,0.095438,0.0841,0.086181,0.0756206,0.0705885,0.0635934,0.0541261]
# drawU(x,BCC,J,R,B,BC)
# plt.subplot(3,3,9)
# J= [0.1768,0.16093,0.13561,0.1375,0.12687,0.13099,0.120651061,0.1106863,0.1005740,0.086322,0.0757108]
# R=[0.1395,0.12805,0.11901,0.1081211,0.094036,0.0845,0.080450917,0.073040859,0.063787,0.0530891,0.044698]
# B= [0.207446,0.178634,0.14973050,0.14643256,0.138743317,0.1150922,0.1100513,0.1072339,0.095060710,0.080163845,0.07063916]
# BCC=[0.254083,0.221760482,0.1715734,0.1512168073,0.131128238,0.101,0.0905681,0.08064906,0.07061085,0.0646334,0.050158532]
# BC = [0.2128,0.184082,0.164434,0.144973,0.115438,0.0985,0.088181,0.0776206,0.0705885,0.0635934,0.0561261]
# drawU(x,BCC,J,R,B,BC)
# plt.show()

# x = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]
# fig, ax = plt.subplots(nrows=3,ncols=3)
# plt.subplot(3,3,1)
# BC = [0.08193786847244789, 0.10325501088506871, 0.17568697444201166, 0.2574680095454631, 0.33386073024187325, 0.4034025934342951, 0.457573694448601, 0.4941962888441762, 0.5470816005914425, 0.6342214685933352]
# J = [0.08175297059521425, 0.13341791156820054, 0.17791230426702498, 0.24470583814051383, 0.2885256509012479, 0.3367666100108856, 0.36392862486907076, 0.3991340636411059, 0.4210211625542926, 0.4581458271896196]
# R = [0.0400459963129941, 0.1054162600405375, 0.19095810256315915, 0.22719329562750157, 0.27493837621321826, 0.3133319328533506, 0.40599281759688766, 0.49347939488784565, 0.5235560484243601, 0.5941094072544971]
# B = [0.09193786847244789, 0.09325501088506871, 0.19568697444201166, 0.2474680095454631, 0.34386073024187325, 0.4134025934342951, 0.477573694448601, 0.5141962888441762, 0.5770816005914425, 0.6642214685933352]
# BCC = [0.07052767992406961, 0.1260415884693341, 0.22956188934071808, 0.2881301790713843, 0.3607302418733631, 0.45583713690965, 0.5656142450995062, 0.5874506297041509, 0.6780519360502726, 0.7178929519315835]
# draw(x, BCC, J, R, B, BC, 'Email')
# plt.legend(loc='upper left', fontsize=14)
# plt.subplot(3,3,2)
# BC = [0.736189103856632774, 0.78368497364657007, 0.8333906650676181, 0.88468568880963, 0.9247193691281701, 0.9485083208150043, 0.954276976781707, 0.9694649644284183, 0.9743294265712263, 0.9756000881726224]
# J = [0.2170209208313259, 0.2585927856261143, 0.29244644650500664, 0.3098707088720463, 0.35184491067516916, 0.3827004359803687, 0.41462913800437085, 0.5160267030503849, 0.5723619523322583, 0.6330451394841902]
# R = [0.22531745173061606, 0.2841056488429961, 0.3314144752878772, 0.39314905632337642, 0.44753930779340905, 0.5123058027959503, 0.5414108449815926, 0.5707094825065783, 0.6167877369621648, 0.6665862499693845]
# B = [0.38977426747582836, 0.47501694848835152, 0.57907391698598862, 0.68476643384356267, 0.7724163948790342, 0.8490876217784672, 0.88075150641107, 0.9052392554331937, 0.92762349563439, 0.9385157608562541]
# BCC = [0.436189103856632774, 0.69368497364657007, 0.8633906650676181, 0.89468568880963, 0.9347193691281701, 0.9585083208150043, 0.964276976781707, 0.968649644284183, 0.9783294265712263, 0.980000881726224]
# draw(x, BCC, J, R, B, BC,'powergrid')
# plt.subplot(3,3,3)
# BC = [0.07478632862500278, 0.11402261442182648, 0.17311338933550126, 0.2486451098844269, 0.2739549391841565, 0.34276130429388344, 0.39357370183936905, 0.4466796405601699, 0.49, 0.526024388029895]
# J = [0.06478632862500278, 0.10402261442182648, 0.16311338933550126, 0.2386451098844269, 0.2839549391841565, 0.33276130429388344, 0.38357370183936905, 0.4566796405601699, 0.50, 0.536024388029895]
# R = [0.04445592209172644, 0.09519500566501138, 0.1435728479812122, 0.20158484870540553, 0.2484958050868284, 0.2834981934935692, 0.32941275988817366, 0.3621058518557707, 0.4087057762284805, 0.4446541498986354]
# B = [0.08157613048587875, 0.10397767453178108, 0.18545072712492106, 0.24059652093301198, 0.30354232911258405, 0.3651586755544527, 0.4162079939914195, 0.4679935306876668, 0.49687719781059836, 0.5684400435716663]
# BCC = [0.10496807467457125, 0.1532999262407896, 0.20907792129587814, 0.26697357180890784, 0.33169789063356597, 0.4037476831226056, 0.4492595866066314, 0.5070069605568445, 0.5416613154756445, 0.6020629973675452]
# draw(x, BCC, J, R, B, BC, 'Jazz')
# plt.subplot(3, 3, 4)
# BC = [0.10890916992611899, 0.16410597154783205, 0.2954310008242591, 0.4055374592833877, 0.453944339820046, 0.4899098260022861, 0.5223219538283269, 0.5584116899618806, 0.58478144592047, 0.6108190940005878]
# J = [0.043215993046501331, 0.0607283414260167, 0.1319921044640145, 0.18931345527436734, 0.2477505754341912, 0.3285635663181069, 0.36746947143239156, 0.41719610334603985, 0.473365776667105, 0.5484277257651455]
# R = [0.0890916992611899, 0.09510597154783205, 0.1854310008242591, 0.2255374592833877, 0.253944339820046, 0.3099098260022861, 0.3523219538283269, 0.3984116899618806, 0.45478144592047, 0.5108190940005878]
# B = [0.06597131681877444, 0.1031903910043446, 0.21337338944080522, 0.29975778835713685, 0.3660598451558904, 0.3951145167435758, 0.4506194848025671, 0.5030072003388395, 0.5546494804682363, 0.6067844997690919]
# BCC = [0.11964363320295524, 0.1897190212113476, 0.3497921999045595, 0.45886578134135136, 0.4976354885959404, 0.5433233140002541, 0.5638773509225422, 0.5914146548072851, 0.631556841597615, 0.6745274780637307]
# draw(x, BCC, J, R, B, BC,'lesmis')
# plt.subplot(3,3,5)
# J = [0.05170209208313259, 0.1085927856261143, 0.19244644650500664, 0.2298707088720463, 0.30184491067516916, 0.3627004359803687, 0.41462913800437085, 0.5160267030503849, 0.5723619523322583, 0.6330451394841902]
# BC = [0.03531745173061606, 0.0941056488429961, 0.1414144752878772, 0.21314905632337642, 0.25753930779340905, 0.3323058027959503, 0.4114108449815926, 0.4707094825065783, 0.5467877369621648, 0.6165862499693845]
# B = [0.04977426747582836, 0.10501694848835152, 0.17907391698598862, 0.24476643384356267, 0.3224163948790342, 0.3790876217784672, 0.4407515510641107, 0.4952392554331937, 0.54762349563439, 0.5985157608562541]
# BCC = [0.046189103856632774, 0.14368497364657007, 0.2533906650676181, 0.33468568880963, 0.4547193691281701, 0.5485083208150043, 0.604276976781707, 0.6394649644284183, 0.6743294265712263, 0.716000881726224]
# R = [0.034439378677834855, 0.07688311688311681, 0.13292098880115617, 0.2012228113840493, 0.22759029500964988, 0.2881397744244734, 0.33252305700879134, 0.3975901266645015, 0.4676679841897233, 0.5122343451057448]
# draw(x, BCC, J, R, B, BC, 'Innovation')
# plt.subplot(3,3,6)
# BC = [0.456267281105992, 0.50221198156682, 0.553917050691247, 0.607142857142857, 0.653225806451613, 0.70603686635945, 0.752534562211982, 0.803225806451614, 0.82811059907834, 0.836866359447002]
# J = [0.06361075181175468, 0.09927713326329424, 0.1321328847325616, 0.17138946485047195, 0.23404442363252425, 0.2705780858216846, 0.301415723958253103, 0.33346932554078236, 0.3703976092165409, 0.4177812389629021]
# R = [0.047078778106561332, 0.10450919207825867, 0.1438458685458448, 0.19260874514276527, 0.25351863795799723, 0.2825600203700269, 0.3283963358147047, 0.3497651782720222, 0.3803705537528403, 0.4426777288093206]
# B = [0.31068704927618428, 0.35695157218475115, 0.39008477494900274, 0.4681777195492396, 0.5295251937128367, 0.58980501741378346, 0.63142648414687544, 0.6903045017659535, 0.7589960300969929, 0.795253252721844]
# BCC = [0.54617750766341232, 0.58923719302804334, 0.6397022279916287, 0.6734731264633082, 0.71864246397427174, 0.7457222943900429, 0.78730024887062027, 0.815665640337774, 0.8355560074623567, 0.8539249619939936]
# draw(x, BCC, J, R, B, BC, 'router')
# plt.subplot(3,3,7)
# J = [0.08514022159783707, 0.10348381776953197, 0.1794395417247252, 0.21258521804999453, 0.22894954507857723, 0.39774424473399594, 0.37133919421821904, 0.3937425571072859, 0.45791047964961007, 0.5150214592274678]
# BC = [0.094439378677834855, 0.13688311688311681, 0.19292098880115617, 0.1912228113840493, 0.24759029500964988, 0.3381397744244734, 0.36252305700879134, 0.4175901266645015, 0.4776679841897233, 0.5022343451057448]
# B = [0.10168053862058002, 0.14459905174190884, 0.19296072663467, 0.24369656963532083, 0.2523848910945684, 0.37503218828861307, 0.3798069144059112, 0.44397531666125367, 0.5092404657622049, 0.5161590568281711]
# BCC = [0.12204368340136776, 0.1517645846217274, 0.23636269804407295, 0.2632290877610649, 0.2989247311827957, 0.3998042952052324, 0.3940456286068713, 0.43791274223232657, 0.49364384146992846, 0.5303790268369616]
# R = [0.044439378677834855, 0.08688311688311681, 0.16292098880115617, 0.1712228113840493, 0.18759029500964988, 0.2681397744244734, 0.31252305700879134, 0.375901266645015, 0.4276679841897233, 0.4722343451057448]
# draw(x, BCC, J, R, B, BC, 'train')
# plt.subplot(3,3,8)
# J = [0.08175297059521425, 0.09341791156820054, 0.17791230426702498, 0.24470583814051383, 0.2685256509012479, 0.3667666100108856, 0.39392862486907076, 0.4791340636411059, 0.5510211625542926, 0.5981458271896196]
# BC = [0.0300459963129941, 0.084162600405375, 0.18095810256315915, 0.21719329562750157, 0.26493837621321826, 0.3033319328533506, 0.38599281759688766, 0.47347939488784565, 0.5035560484243601, 0.5741094072544971]
# B = [0.09193786847244789, 0.09225501088506871, 0.19568697444201166, 0.2474680095454631, 0.34386073024187325, 0.4134025934342951, 0.477573694448601, 0.5141962888441762, 0.5770816005914425, 0.6642214685933352]
# BCC = [0.07052767992406961, 0.1260415884693341, 0.22956188934071808, 0.2881301790713843, 0.3607302418733631, 0.45583713690965, 0.5656142450995062, 0.5874506297041509, 0.6780519360502726, 0.7178929519315835]
# R = [0.0400459963129941, 0.0854162600405375, 0.16095810256315915, 0.20719329562750157, 0.27493837621321826, 0.3133319328533506, 0.36599281759688766, 0.45347939488784565, 0.5035560484243601, 0.5641094072544971]
# draw(x, BCC, J, R, B, BC, 'Highschool')
# plt.subplot(3,3,9)
# J = [0.06361075181175468, 0.1309927713326329424, 0.1721328847325616, 0.22138946485047195, 0.28404442363252425, 0.3505780858216846, 0.41415723958253103, 0.47346932554078236, 0.5303976092165409, 0.5777812389629021]
# BC = [0.047078778106561332, 0.09450919207825867, 0.1538458685458448, 0.19260874514276527, 0.26351863795799723, 0.3625600203700269, 0.4383963358147047, 0.5297651782720222, 0.5803705537528403, 0.6326777288093206]
# B = [0.05068704927618428, 0.11695157218475115, 0.15008477494900274, 0.2181777195492396, 0.2795251937128367, 0.35980501741378346, 0.41142648414687544, 0.4803045017659535, 0.5489960300969929, 0.6195253252721844]
# BCC = [0.06617750766341232, 0.101923719302804334, 0.1697022279916287, 0.2334731264633082, 0.29864246397427174, 0.3957222943900429, 0.46730024887062027, 0.555665640337774, 0.6155560074623567, 0.6939249619939936]
# R = [0.038078778106561332, 0.08450919207825867, 0.1238458685458448, 0.16260874514276527, 0.22351863795799723, 0.3225600203700269, 0.3883963358147047, 0.4697651782720222, 0.5203705537528403, 0.5826777288093206]
# draw(x, BCC, J, R, B, BC, 'Oz')
# plt.show()

