import networkx as nx
import math
import time

def constructG(fileName):
    G = nx.DiGraph().to_undirected()
    with open('networks/'+fileName, 'r') as f:
        for position, line in enumerate(f):
            t = line.strip().split(',')
            G.add_edge(t[0], t[1], weight = 1)
    return G

def calculateQ(edgeInCommunity, allEdge, allDegreeInCommunity):
    Q = float(edgeInCommunity)/allEdge - (float(allDegreeInCommunity)/(2*allEdge))**2
    return Q

def divideCommunity(G):
    degreeDict = {}
    communityNodeDict = {}
    nodeCommunityDict = {}
    edgesDict = {}
    edgesInCommunityDict = {}
    degreesInCommunityDict = {}
    m = 0
    for edge in G.edges():
        weight = G.get_edge_data(edge[0], edge[1])['weight']
        m += weight
        edgesDict[edge] = weight
        edgesDict[(edge[1], edge[0])] = weight

    for node in G.nodes():
        degreeDict[node] = 0
        for neighbor in G.neighbors(node):
            if node == neighbor:
                degreeDict[node] += 2*edgesDict[(node, neighbor)]
            else:
                degreeDict[node] += edgesDict[(node, neighbor)]
    for node in G.nodes():
        communityNodeDict[node] = [node]
        nodeCommunityDict[node] = node
        if G.has_edge(node, node):   #have self-loop
            edgesInCommunityDict[node] = edgesDict[(node, node)]
        else:
            edgesInCommunityDict[node] = 0
        degreesInCommunityDict[node] = degreeDict[node]
    # nodes = sorted(G.degree().iteritems(), key=lambda x: x[1])
    # nodes = [x[0] for x in nodes]
    while True:
        isMove = 0
        for node in G.nodes():
            c1 = nodeCommunityDict[node]
            kIn = {}
            for neighbor in G.neighbors(node):
                tempC = nodeCommunityDict[neighbor]
                if tempC not in kIn.keys():
                    kIn[tempC] = edgesDict[(node, neighbor)]
                else:
                    kIn[tempC] += edgesDict[(node, neighbor)]
            if c1 not in kIn.keys():
                kIn[c1] = 0
            maxQ = 0
            bestCommunity = c1
            for community in kIn.keys():
                if community != c1:
                    beforeQ = (calculateQ(edgesInCommunityDict[c1], m, degreesInCommunityDict[c1]) +
                               calculateQ(edgesInCommunityDict[community], m, degreesInCommunityDict[community]))
                    if G.has_edge(node, node):
                        nowQ = (calculateQ(edgesInCommunityDict[c1]-kIn[c1], m, degreesInCommunityDict[c1]-degreeDict[node]) +
                                   calculateQ(edgesInCommunityDict[community]+kIn[community]+edgesDict[(node, node)], m, degreesInCommunityDict[community]+degreeDict[node]))
                    else:
                        nowQ = (calculateQ(edgesInCommunityDict[c1] - kIn[c1], m,
                                           degreesInCommunityDict[c1] - degreeDict[node]) +
                                calculateQ(edgesInCommunityDict[community] + kIn[community], m,
                                           degreesInCommunityDict[community] + degreeDict[node]))

                    detaQ = nowQ - beforeQ
                    if detaQ > maxQ:
                        maxQ = detaQ
                        bestCommunity = community
            if maxQ != 0:
                isMove += 1
                communityNodeDict[bestCommunity].append(node)
                communityNodeDict[c1].remove(node)
                nodeCommunityDict[node] = bestCommunity
                edgesInCommunityDict[bestCommunity] += kIn[bestCommunity]
                if G.has_edge(node, node):
                    edgesInCommunityDict[bestCommunity] += edgesDict[(node, node)]
                edgesInCommunityDict[c1] -= kIn[c1]
                degreesInCommunityDict[bestCommunity] += degreeDict[node]
                degreesInCommunityDict[c1] -= degreeDict[node]
        if isMove == 0:
            break
    allQ = 0
    for community in communityNodeDict.keys():
        if len(communityNodeDict[community]) == 0:
            communityNodeDict.pop(community)
        else:
            allQ += calculateQ(edgesInCommunityDict[community], m, degreesInCommunityDict[community])
    return communityNodeDict, allQ, edgesDict, edgesInCommunityDict, nodeCommunityDict

def superG(communityNodeDict, edgesDict, edgesInCommunityDict, nodeCommunityDict):
    superEdges = {}
    num = 0
    for edge in edgesDict:
        c1 = nodeCommunityDict[edge[0]]
        c2 = nodeCommunityDict[edge[1]]
        if c1 != c2:
            if (c1, c2) in superEdges:
                superEdges[(c1, c2)] += edgesDict[edge]
            elif (c2, c1) in superEdges:
                superEdges[(c2, c1)] += edgesDict[edge]
            else:
                superEdges[(c1, c2)] = edgesDict[edge]

    spG = nx.DiGraph().to_undirected()
    supNodeDict = {}
    for community in communityNodeDict.keys():
        supNodeDict[community] = communityNodeDict[community]
    nodes = supNodeDict.keys()[:]
    for edge in superEdges.keys():
        spG.add_edge(edge[0], edge[1], weight = superEdges[edge]/2)
    for i in edgesInCommunityDict.keys():
        if edgesInCommunityDict[i] != 0:
            spG.add_edge(i, i, weight = edgesInCommunityDict[i])
    return spG

def findRealCommunity(communityNodeDict,realcommunity):
    communitys = {}
    for i in communityNodeDict.keys():
        communitys[i] = []
        for node in communityNodeDict[i]:
            communitys[i].extend(realcommunity[node])
    return communitys

def mainFunc(G):
    bestcommunity, maxQ, edgesDict, edgesInCommunityDict, nodeCommunityDict = divideCommunity(G)
    realcommunity = bestcommunity
    n = 1
    print 'have finish', n
    while True:
        n = n + 1
        spG = superG(bestcommunity, edgesDict, edgesInCommunityDict, nodeCommunityDict)
        print 'spG'
        communityNodeDict, allQ, edgesDict, edgesInCommunityDict, nodeCommunityDict = divideCommunity(spG)
        print 'have finish', n
        if allQ > maxQ+0.000001:
            maxQ = allQ
            bestcommunity = communityNodeDict
            realcommunity = findRealCommunity(communityNodeDict, realcommunity)
            print 'real'
        else:
            break
    return realcommunity


G = constructG('model4.csv')
print 'construct network finished'
begin = time.clock()
realcommunity = mainFunc(G)
print len(max(realcommunity.iteritems(), key=lambda x: len(x[1]))[1])
print time.clock() - begin

output = open('result/real2Result', 'w')
n = 0
for community in realcommunity.keys():
    n += 1
    output.write('c'+str(n))
    for node in realcommunity[community]:
        output.write(','+node)
    output.write('\n')
output.close()
