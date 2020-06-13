import ahocorasick
import Levenshtein
import re
import numpy as np

from buildGraphy import QAGraph
import torch
from model.LSTM.tool import load_Params
from model.LSTM.train_lstm import predict
from model.LSTM.tool import load_symptom_vector
from model.LSTM.tool import load_disease_vector
from utils.util import cosine_similarity


class EntityExtractor:
    def __init__(self, graph: QAGraph):
        self.graph = graph
        self.alias_tree = self.build_actree(self.graph.aliasSet)
        self.check_tree = self.build_actree(self.graph.checkSet)
        self.department_tree = self.build_actree(self.graph.departmentSet)
        self.disease_tree = self.build_actree(self.graph.diseaseSet)
        self.medical_tree = self.build_actree(self.graph.medicalSet)
        self.position_tree = self.build_actree(self.graph.positionSet)
        self.symptom_tree = self.build_actree(self.graph.symptomsSet)

        # intention 分类模型
        # self.tfidfmodel = None    # idf 的模型(效果不好)
        self.result = {}
        self.PATH = "E:/Workspaces/Python/KG/QA_healty39/data/output"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.QUEST_vocab, self.INTENT_vocab = load_Params(self.PATH, self.device)
        self.symptom_vectors = load_symptom_vector()
        self.disease_vectors = load_disease_vector()

        self.symptom_stop_words = ["怎么", "办", "问题", "情况", "毛病", "引起", "什么", "由",
                                    "导致", "回事", "哪种病", "病", "症状", "表现", "有", "整",
                                    "搞", "才行","描述","东西","简介","介绍","疾病","大概率","概率",
                                    "可能", "可能性", "哪个科", "什么科", "科室", "挂", "啥", "治好",
                                    "治愈", "医治", "医", "属于", "会", "有", "现象", "是", "检查", "哪个", "科",
                                    "康复", "得了", "应该", "咋", "治疗", "治", "痊愈", "好", "吗"]
        self.intention = ""

    def predict(self, questionList):
        """
        用模型预测
        :param questionList:
        :return:
        """
        intentions = predict(self.QUEST_vocab, self.INTENT_vocab, self.model, questionList)
        return intentions

    def getFeatures(self, questions):
        """
        计算特征
        :param questions:
        :return:
        """
        tfidf = self.getTFIDF(questions)
        return tfidf

    def getSimEntity(self, input, wordlist):
        """
        1. 用编辑距离筛选
        2. 用词向量计算相似度
        :param input:
        :param wordlist:
        :return:
        """
        simList = []
        for word in wordlist:
            sim = Levenshtein.distance(word, input)
            simList.append((word, sim))
        simList.sort(key=lambda x: x[1], reverse=False)
        smallestsim = simList[0][1]
        simword = [k for k, v in simList if v == smallestsim]
        return simword

    def getEneityBySim(self, input, wordListA, wordListB):
        """
        1. 找到编辑距离最近的实体
        2. 第一个字相同
        3. 用词向量比较相似度

        :param input:
        :param wordList:
        :return:
        """

        def getVectors(symptom_vec, input):
            vector = [0.0,] * 300
            vector = np.array(vector)
            for char in input:
                s = symptom_vec.vectors[symptom_vec.stoi[char]]
                s = s.data.numpy()
                vector += s
            return vector

        # 1. 编辑距离
        simwordA = self.getSimEntity(input, wordListA)
        simwordB = self.getSimEntity(input, wordListB)

        # 2. 开头第一个字相同的元素
        simword_A2 = []
        for word in simwordA:
            if word[0] == input[0]:
                simword_A2.append(word)

        simword_B2 = []
        for word in simwordB:
            if word[0] == input[0]:
                simword_B2.append(word)

        # 3. 词向量比较相似度
        symptom_vec = load_symptom_vector()
        disease_vec = load_disease_vector()

        input_vector_A = getVectors(symptom_vec, input)
        input_vector_B = getVectors(disease_vec, input)
        simword_A2_vector = [getVectors(symptom_vec, word) for word in simword_A2]
        simword_B2_vector = [getVectors(symptom_vec, word) for word in simword_B2]

        simListA = []
        for vec in simword_A2_vector:
            sim = cosine_similarity(input_vector_A, vec)
            simListA.append(sim)

        simListB = []
        for vec in simword_B2_vector:
            sim = cosine_similarity(input_vector_B, vec)
            simListB.append(sim)

        finalA = np.argmax(simListA)
        finalB = np.argmax(simListB)

        if simListA[finalA] > simListB[finalB]:
            return "Symptom", simword_A2[finalA]
        else:
            return "Disease", simword_B2[finalB]

    def build_actree(self, wordlist):
        """
        构造actree，加速过滤
        """
        actree = ahocorasick.Automaton()
        for index, word in enumerate(wordlist):
            actree.add_word(word, (index, word))
        actree.make_automaton()
        return actree

    def getLongestEntity(self, tree, input):
        """
        用ac自动机， 找出最长的实体
        :param tree:
        :param input:
        :return:
        """
        res = []
        long_word = ""
        for word in tree.iter(input):
            if len(word[1][1]) > len(long_word):
                long_word = word[1][1]
        if long_word == "":
            return None
        res.append(long_word)
        return res

    def removeUnrelatedWords(self, input, stopwords):
        for stop in stopwords:
            input = input.replace(stop, "")
        return input

    def extractEntities(self, input):
        """
        提取实体
        TODO 找不到实体，需要找最相似的实体
        :param input:
        :return:
        """
        self.result = {}
        temp = self.getLongestEntity(self.alias_tree, input)
        if temp:
            self.result["Alias"] = temp

        temp = self.getLongestEntity(self.check_tree, input)
        if temp:
            self.result["Check"] = temp

        temp = self.getLongestEntity(self.department_tree, input)
        if temp:
            self.result["Department"] = temp

        temp = self.getLongestEntity(self.disease_tree, input)
        if temp:
            self.result["Disease"] = temp

        temp = self.getLongestEntity(self.medical_tree, input)
        if temp:
            self.result["Medical"] = temp

        temp = self.getLongestEntity(self.position_tree, input)
        if temp:
            self.result["Position"] = temp

        temp = self.getLongestEntity(self.symptom_tree, input)
        if temp:
            self.result["Symptom"] = temp

        # 没有获取到实体
        if self.result == {}:
            # # 如果intention == 查询疾病
            # if self.intention == "query_disease":
            #     word = self.removeStopWords(input, self.symptom_stop_words)
            #     symptom = self.getEneityBySim(word, self.graph.symptomsSet)
            #     self.result["Symptom"] = symptom
            word = self.removeUnrelatedWords(input, self.symptom_stop_words)
            type, entity = self.getEneityBySim(word, self.graph.symptomsSet, self.graph.diseaseSet)
            self.result[type] = [entity]
            return type, word
        return None

    def removeDiseaseNameFromQuestion(self, inputs):
        """
        从问句中删除疾病的名称，减少OOV（把diseaseName替换成#）
        :param inputs: 字符串列表
        :return:
        """
        res = []
        for input in inputs:
            long_prefix = ""
            for word in self.disease_tree.iter(input):
                if len(word[1][1]) > len(long_prefix):
                    long_prefix = word[1][1]
            if long_prefix != "":
                input = input.replace(long_prefix, "#")
            res.append(input)
        return res

    def removeSymptomFromQuestion(self, inputs):
        """
        从问句中删除疾病的名称，减少OOV（把diseaseName替换成#）
        :param inputs: 字符串列表
        :return:
        """
        res = []
        for input in inputs:
            long_prefix = ""
            for word in self.symptom_tree.iter(input):
                if len(word[1][1]) > len(long_prefix):
                    long_prefix = word[1][1]
            if long_prefix != "":
                input = input.replace(long_prefix, "@")
            res.append(input)
        return res

    def removeStopWords(self, inputs):
        """
        去除停用词（目前就是符号）
        :param inputs:
        :return:
        """
        res = []
        for input in inputs:
            input = re.sub("[?？！!]", "", input)
            res.append(input)
        return res

    def extractor(self, input):
        # 1. 提取实体
        res = self.extractEntities(input)
        if res is not None:
            type, word = res
            if type=="Disease":
                input = input.replace(word, "#")
            elif type=="Symptom":
                input = input.replace(word, "@")

        # 2. 提取意图
        # 预处理
        input = self.removeDiseaseNameFromQuestion([input])
        input = self.removeSymptomFromQuestion(input)
        input = self.removeStopWords(input)

        # print(input)

        intention = self.predict(input)
        self.result["intention"] = intention
        return self.result


if __name__ == "__main__":
    from QA import qagraph
    e = EntityExtractor(qagraph)
    result = e.extractor(["#会死吗"])
    # print(result)
    # res = e.getEneityBySim("头疼是什么病", qagraph.symptomsSet)
    # e.symptom_tree.iter("头疼")
    print(result)