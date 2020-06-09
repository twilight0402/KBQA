import ahocorasick
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

from buildGraphy import QAGraph
import torch
from model.tool import load_Params
from model.train_lstm import predict


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

    def predict(self, questionList):
        intentions = predict(self.QUEST_vocab, self.INTENT_vocab, self.model, questionList)
        return intentions

    def getCutRes(self, questions):
        """
        二维句子列表，用来分词
        :param questions:
        :return:
        """
        questions_ = []
        for sentence in questions:
            sentence = " ".join(list(jieba.cut(sentence)))
            questions_.append(sentence)
        return questions_

    def getTFIDF(self, questions):
        """
        二维句子列表，用来分词，然后再计算 tf-idf
        :param questions:
        :return:
        """
        corpus = self.getCutRes(questions)
        # 转换成一维
        # corpus = [sent for intention in corpus for sent in intention]

        self.tfidfmodel = TfidfVectorizer()
        self.tfidfmodel.fit(corpus)
        tfidf_matrix = self.tfidfmodel.transform(corpus)
        return tfidf_matrix

    def getFeatures(self, questions):
        tfidf = self.getTFIDF(questions)
        return tfidf

    def build_actree(self, wordlist):
        """
        构造actree，加速过滤
        """
        actree = ahocorasick.Automaton()
        for index, word in enumerate(wordlist):
            actree.add_word(word, (index, word))
        actree.make_automaton()
        return actree

    def extractEntities(self, input):
        """
        提取实体
        TODO 找不到实体，需要找最相似的实体
        :param input:
        :return:
        """
        for word in self.alias_tree.iter(input):
            if self.result.get("Alias") is None:
                self.result["Alias"] = [word[1][1]]
            else:
                self.result["Alias"].append(word[1][1])
        for word in self.check_tree.iter(input):
            if self.result.get("Check") is None:
                self.result["Check"] = [word[1][1]]
            else:
                self.result["Check"].append(word[1][1])
        for word in self.department_tree.iter(input):
            if self.result.get("Department") is None:
                self.result["Department"] = [word[1][1]]
            else:
                self.result["Department"].append(word[1][1])
        for word in self.disease_tree.iter(input):
            if self.result.get("Disease") is None:
                self.result["Disease"] = [word[1][1]]
            else:
                self.result["Disease"].append(word[1][1])
        for word in self.medical_tree.iter(input):
            if self.result.get("Medical") is None:
                self.result["Medical"] = [word[1][1]]
            else:
                self.result["Medical"].append(word[1][1])
        for word in self.position_tree.iter(input):
            if self.result.get("Position") is None:
                self.result["Position"] = [word[1][1]]
            else:
                self.result["Position"].append(word[1][1])
        for word in self.symptom_tree.iter(input):
            if self.result.get("Symptom") is None:
                self.result["Symptom"] = [word[1][1]]
            else:
                self.result["Symptom"].append(word[1][1])

    def removeDiseaseNameFromQuestion(self, inputs):
        """
        从问句中删除疾病的名称，减少OOV（把diseaseName替换成#）
        :param inputs: 字符串列表
        :return:
        """
        res = []
        for input in inputs:
            for word in self.disease_tree.iter(input):
                input = input.replace(word[1][1], "#")
            res.append(input)
        return res

    def extractor(self, input):
        # 1. 提取实体
        self.extractEntities(input)

        # 2. 提取意图
        intention = self.predict(self.removeDiseaseNameFromQuestion([input]))
        self.result["intention"] = intention
        return self.result


if __name__ == "__main__":
    e = EntityExtractor()
    result = e.extractor("哮喘病怎么治疗")
    print(result)
