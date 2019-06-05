#coding=utf-8

import networkx as nx
import xlrd
import json
import re
import nltk
import Levenshtein

def nameSimilarity(n1, n2):
    n1 = n1.upper()
    n2 = n2.upper()
    if n1 == n2 or n1.replace(',', '') == n2.replace(',', ''):
        return 2
    else:
        if ', ' in n1 and ', ' in n2:
            surname1, name1= n1.split(', ', 1)
            surname2, name2 = n2.split(', ', 1)
            if surname1 != surname2:
                return 0
            else:
                Min = min(len(name1), len(name2))
                if name1 == name2[:Min]:
                    return 1
                elif name2 == name1[:Min]:
                    return 2
                else:
                    return 0
        else:
            return 0

def nameSimilarity2(n1, n2):
    n1 = n1.upper()
    n2 = n2.upper()
    if n1 == n2 or n1.replace(',', '') == n2.replace(',', ''):
        return True
    else:
        try:
            surname1, name1 = n1.replace(',', '').split(' ', 1)
            surname2, name2 = n2.replace(',', '').split(' ', 1)
        except:
            return False
        if surname1 != surname2:
            return False
        else:
            name1 = name1.split(' ')
            name2 = name2.split(' ')

            for i in range(min(len(name1), len(name2))):
                if len(name1[i]) == len(name2[i]):
                    if name1[i] != name2[i]:
                        return False
                else:
                    if min(len(name1[i]), len(name2[i])) > 1:
                        return False
                    else:
                        if name1[i][:1] != name2[i][:1]:
                            return False

            return True


def MatchName(s, array):
    if s == '':
        return ''

    maxRatio = 0
    best = ''
    for i in array:
        tempRatio = Levenshtein.ratio(s, i.upper())
        if tempRatio > maxRatio:
            maxRatio = tempRatio
            best = i
    if maxRatio>0.6:

        return best
    else:
        return ''

def changeBlocks(author, keys):
    author = author.upper()
    up = ''
    down = []

    for i in keys:
        if nameSimilarity(author, i) == 1:
            down.append(i)
        elif nameSimilarity(author, i) == 2:
            up = i
        else:
            pass
    return up, down

def nameProduct(s):
    if s == '':
        return ''
    if ',' in s:
        try:
            surname, name = s.split(', ', 1)
            return surname.replace('. ', '').replace('-', '').replace('.', '').replace(' ', '') + ', '+ \
                    name.replace('. ', '').replace('-', '').replace('.', '')
        except:
            return s
    else:
        return s


def makeData(inputFile, outputFileDoc):
    # ExcelFile = xlrd.open_workbook(inputFile)
    # sheet = ExcelFile.sheet_by_name(ExcelFile.sheet_names()[0])
    # ncols = sheet.ncols
    # nrows = sheet.nrows
    PaperESI= {}
    Blocks = {}  # DOI(DI):作者（AU，AF），基金号（FU），邮箱(EM),摘要(AB)，关键词(DE),工作单位（C1）,ESI被引用,编号(UT)，RI，OI

    with open(inputFile, 'r') as f:
        index = -1
        for i in f.readlines():
            index += 1
            if index==0:
                continue
            print index
            context = i.replace('"', '').strip().split('\t')
            UT = context[62]
            ESI = int(context[31])
            EM = context[24]
            AU = context[1]
            AF = context[5]
            TI = context[8]
            DE = context[19]
            AB = context[21]
            C1 = context[22]
            RI = context[25]
            OI = context[26]
            FU = context[27]

            PaperESI[UT] = ESI
            tempAF = []
            tempFU = []
            tempRI = {}
            tempOI = {}
            tempC1 = {}

            for fu in re.findall('\[(.*?)\]', FU):
                for number in fu.split(', '):
                    if number != '':
                        tempFU.append(number)

            for x in C1.split('; ['):
                temp = x.split('] ')

                if len(temp) > 1:
                    for j in temp[0].replace('[', '').split('; '):

                        name = nameProduct(j).upper()

                        if name in tempC1.keys():
                            tempC1[name].append(temp[1])
                        else:
                            tempC1[name] = [temp[1]]
                else:
                    for author in AF.split('; '):
                        author = nameProduct(author).upper()
                        tempC1[author] = []
                        for c in temp[0].split('; '):
                            tempC1[author].append(c)

            for author in AF.split('; '):
                author = nameProduct(author)
                tempAF.append(author)

            tempAU = []
            for author in AU.split('; '):
                author = nameProduct(author)
                tempAU.append(author)

            for j in sorted(RI.split('; ')):
                name = nameProduct(j.split('/')[0]).upper()
                matchName = MatchName(name, tempAF)
                if matchName != '':
                    if matchName in tempRI.keys():
                        tempRI[matchName].append(j.split('/')[1])
                    else:
                        tempRI[matchName] = [j.split('/')[1]]

            for j in sorted(OI.split('; ')):
                name = nameProduct(j.split('/')[0]).upper()
                matchName = MatchName(name, tempAF)
                if matchName != '':
                    if matchName in tempOI.keys():
                        tempOI[matchName].append(j.split('/')[1])
                    else:
                        tempOI[matchName] = [j.split('/')[1]]

            N = 0
            for author in AU.split('; '):
                author = nameProduct(author)

                tempArray = {}  # AU,AF,UT,RI,OI,C1,FU,EM,TI,AB,DE,reference,citation
                tempArray['AU'] = author
                tempArray['AF'] = tempAF[N]
                tempArray['UT'] = UT
                tempArray['coAuthors'] = tempAU

                if tempAF[N] in tempRI.keys():
                    tempArray['RI'] = tempRI[tempAF[N]]
                else:
                    tempArray['RI'] = []

                if tempAF[N] in tempOI.keys():
                    tempArray['OI'] = tempOI[tempAF[N]]
                else:
                    tempArray['OI'] = []

                if tempAF[N].upper() in tempC1.keys():
                    tempArray['C1'] = tempC1[tempAF[N].upper()]
                else:
                    tempArray['C1'] = []

                tempArray['FU'] = tempFU
                tempArray['EM'] = EM.split('; ')
                tempArray['TI'] = TI
                tempArray['AB'] = AB
                tempArray['DE'] = DE

                up, down = changeBlocks(author, Blocks.keys())

                if up == '':
                    Blocks[author.upper()] = [tempArray]
                    for key in down:
                        Blocks[author.upper()] = Blocks[author.upper()] + Blocks[key]
                        del Blocks[key]
                else:
                    Blocks[up].append(tempArray)
                N += 1

    f = open(outputFileDoc+'Blocks', 'w')
    f.write(json.dumps(Blocks))
    f.close()

    f = open(outputFileDoc + 'PaperESI', 'w')
    f.write(json.dumps(PaperESI))
    f.close()

def divided(data, AuthorsNew, Edges = []):
    G = nx.DiGraph().to_undirected()
    for i in range(len(data)):
        G.add_node(i, d=data[i])

##第一轮根据确定信息进行信息补全
    for i in range(len(data) - 1):
        for j in range(i+1, len(data)):
            if data[i]['UT'] == data[j]['UT']:
                continue

            if len(set(data[i]['RI']) & set(data[j]['RI'])) >= 1:
                G.add_edge(i, j)
                continue

            if len(set(data[i]['OI']) & set(data[j]['OI'])) >= 1:
                G.add_edge(i, j)
                continue

            if data[i]['AF'] != data[i]['AU'] and data[j]['AF'] != data[j]['AU']:
                if data[i]['AF'].upper() == data[j]['AF'].upper() and len(set([x.upper() for x in data[i]['EM']]) \
                                                                                  & set([x.upper() for x in data[j]['EM']])) >= 1:
                    # print data[i]['C1'][0], data[j]['C1'][0],data[i]['UT'],data[j]['UT']
                    G.add_edge(i, j)

    for i in list(nx.connected_components(G)):
        I = list(i)
        allName = []
        allUT = []
        for j in I:
            allName.append(data[j]['AF'])
            allUT.append(data[j]['UT'])

        tempName = max(allName, key=lambda x:len(x))
        for j in I:
            data[j]['AF'] = tempName
            data[j]['allUT'] = set(allUT)

##第二轮根据相似度进行连边

    for i in range(len(data) - 1):
        # if data[i]['AF'] != data[i]['AU']:#无全称数据，连接最大概率
        for j in range(i+1, len(data)):
            pMax = 0.8
            if G.has_edge(i, j):
                continue

            if len(data[i]['allUT'] & data[j]['allUT']) >= 1:
                Edges.append((data[i]['AF'] + ';' + data[i]['UT'], data[j]['AF'] + ';' + data[j]['UT'], 0))
                continue


            pName = 0


            if data[i]['AF'] != data[i]['AU'] and data[j]['AF'] != data[j]['AU']:
                # if nameSimilarity2(data[i]['AF'], data[j]["AF"]) == False:
                #     if data[i]['AU'].upper() == data[j]["AU"].upper() and data[i]['AF'].upper().replace(' ', '') == data[j]["AF"].upper().replace(' ', ''):
                #         pName = 0.4
                #     else:
                #         Edges.append((data[i]['AF'] + ';' + data[i]['UT'], data[j]['AF'] + ';' + data[j]['UT'], 0))
                #         continue
                # else:
                #     pName = 0.4
                if data[i]['AF'].upper() != data[j]["AF"].upper():
                    continue
                else:
                    pName = 0.4
                    if len(set([x.upper() for x in data[i]['EM']]) & set([x.upper() for x in data[j]['EM']])) >= 1:
                        G.add_edge(i, j)
                        continue
                    else:
                        pEM = 0
            else:
                # if nameSimilarity2(data[i]['AU'], data[j]['AU']) == False:
                #     Edges.append((data[i]['AF'] + ';' + data[i]['UT'], data[j]['AF'] + ';' + data[j]['UT'], 0))
                #     continue
                # else:
                #     pMax = 0.95
                if data[i]['AU'].upper() != data[j]["AU"].upper():
                    continue
                else:
                    pMax = 0.9
                    if len(set([x.upper() for x in data[i]['EM']]) & set([x.upper() for x in data[j]['EM']])) >= 1:
                        pEM = 0.4
                    else:
                        pEM = 0




            if data[i]['AU'].upper() == data[j]['AU'].upper():
                pAU = 0.4*(1-10**(-len(set(data[i]['coAuthors']) & set(data[j]['coAuthors']))+1))#共同作者
            else:
                pAU = 0.4 * (1 - 10 ** (-len(set(data[i]['coAuthors']) & set(data[j]['coAuthors']))))

            pC1 = addressSimilarity(data[i]['C1'], data[j]['C1'])*0.8#共同地址

            pTheme = compute_sim(data[i]['TI'], data[j]['TI'], data[i]['AB'], \
                                 data[j]['AB'], data[i]['DE'], data[j]['DE'])*0.4

            pFU = min(0.5, len(set(data[i]['FU']) & set(data[j]['FU'])) * 0.5)



            p = 1-(1-pEM)*(1-pAU)*(1-pC1)*(1-pTheme)*(1-pName)*(1-pFU)
            Edges.append((data[i]['AF'] + ';' + data[i]['UT'], data[j]['AF'] + ';' + data[j]['UT'], p))
            if p >= pMax:
                G.add_edge(i, j)
    for i in list(nx.connected_components(G)):
        I = list(i)
        temp = [max([data[x]['AF'] for x in I], key=lambda x:len(x))]
        for j in I:
            temp.append(data[j]['UT'])
        allD = []
        for j in I:
            allD.append(G.node[j]['d'])
        temp.append(allD)
        AuthorsNew.append(temp)


# 输入获取词性的string，返回只包含动词和名次的词list
def get_nn_vb(string):
    string_list = []
    pos = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']  # 只选择名词、动词
    # text = nltk.word_tokenize("And now for something completely different")
    text = nltk.word_tokenize(string)
    pos_result = nltk.pos_tag(text)  # the output form is tuple
    for tuple in pos_result:
        word = tuple[0]
        pos_word = tuple[1]
        if pos_word in pos:
            string_list.append(word)
    return string_list

# 分别输入title、abstract、keyword，和三者对应的权重，返回相似度数值
def compute_sim(title1,title2,abstract1,abstract2,key_words1,key_words2):
    sim = 0.0
    title_number = len(set(title1) & set(title2))/float(min(len(set(title1)), len(set(title2))) + 1)
    abstract_number = len(set(abstract1) & set(abstract2))/float(min(len(set(abstract1)), len(set(abstract2))) + 1)
    key_words_number = len(set(key_words1) & set(key_words2))/float(min(len(set(key_words2)), len(set(key_words2))) + 1)
    sim = title_number * 0.4 + abstract_number * 0.3 + key_words_number * 0.3
    return sim

def addressSimilarity(ad1, ad2):
    m = 0
    for i in ad1:
        for j in ad2:
            if i.split(', ')[-1] == j.split(', ')[-1]:
                m = max(m, 0.2)
            if i.split(', ')[0] == j.split(', ')[0]:
                m = max(m, 0.6)
            try:
                if i.split(', ')[1] == j.split(', ')[1]:
                    m = max(m, 0.8)
            except:
                pass
            if i == j:
                m = max(m, 1)

    return m

def mainMethod(inputFilePath, outputFilePath):
    try:
        f = open('Blocks', 'r')  # 读取生成文件blocks
        Blocks = json.loads(f.read())
        f.close()

        f = open('PaperESI', 'r')  # 读取生成文件ESI
        PaperESI = json.loads(f.read())
        f.close()
    except:
        makeData(inputFilePath, '')
        f = open('Blocks', 'r')  # 读取生成文件blocks
        Blocks = json.loads(f.read())
        f.close()

        f = open('PaperESI', 'r')  # 读取生成文件ESI
        PaperESI = json.loads(f.read())
        f.close()

    AuthorsNew = []
    for author in Blocks:
        data = Blocks[author]
        if len(data) > 1:
            divided(data, AuthorsNew)
        elif len(data) == 0:
            pass
        else:
            AuthorsNew.append([data[0]['AF'], data[0]['UT'], [data[0]]])

    Blocks2 = {}
    for i in AuthorsNew:
        if i[0].upper() in Blocks2.keys():
            Blocks2[i[0].upper()] = Blocks2[i[0].upper()] + i[-1]
        else:
            Blocks2[i[0].upper()] = i[-1]

    Edges = []
    AuthorsNew2 = []
    for author in Blocks2:
        data = Blocks2[author]
        if len(data) > 1:
            divided(data, AuthorsNew2, Edges)
        elif len(data) == 0:
            pass
        else:
            AuthorsNew2.append([data[0]['AF'], data[0]['UT'], [data[0]]])
    results = []

    for i in AuthorsNew2:
        temp = [i[0], len(i) - 2]
        ESI = 0
        for j in i[1:len(i) - 1]:
            ESI += PaperESI[j]
        temp.append(ESI)
        for j in i[1:len(i) - 1]:
            temp.append(j)
        results.append(temp)

    results = sorted(results, key=lambda x: (x[0], x[1]))

    f = open(outputFilePath+'/Results.txt', 'w')  # 输出最终结果
    for i in results:
        f.write(i[0] + ';' + str(i[1]) + ';' + str(i[2]))
        for j in i[3:]:
            f.write(';' + j)
        f.write('\n')
    f.close()

if __name__ == '__main__':
    mainMethod(unicode('/Users/yuenyu/Desktop/nature-son-chinese2014-2018.txt',"utf-8"), '/Users/yuenyu/Desktop')

