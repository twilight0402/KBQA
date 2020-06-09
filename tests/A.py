import os
import numpy as np
###############读取词向量

def getEmbedding():
    embeddings_index = {}
    f = open(os.path.join(r'E:\Workspaces\Python\KG\QA_healty39\data\embedding', 'sgns.sogou.word'),'r', encoding="utf-8")
    for line in f.readlines():
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

# print('Found %s word vectors.' % len(embeddings_index))


# from py2neo import Graph, Node, Relationship
# graph = Graph("http://localhost:7474", username="neo4j", password="123456")
# sql = "MATCH (d:Disease)-[:NEED_MEDICAL]->(n) WHERE d.name='哮喘' return d.name, n.name"
# res = graph.run(sql).data()
#
# print(res)
# print(type(res))
#
# for item in res:
#     print(item.get("d.name"))
#     print(item["n.name"])
# # print(res)

# a = 11
# assert a == 10
a = {"a": 1}
items = list(a.items())

k, v = items[0]
print(k, v)
print(items[0][0])