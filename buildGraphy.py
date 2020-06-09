from utils.DB import getQueryRes, getConnection
from py2neo import Graph, Node, Relationship
import re


class QAGraph:
    def __init__(self):
        self.graph = Graph("http://localhost:7474", username="neo4j", password="123456")
        self.splitStr = r"[,，、\\]"

        # 实体
        self.diseases = []      # 疾病名称
        self.aliases = []       # 别名
        self.symptoms = []      # 症状
        self.position = []      # 部位
        self.departments = []   # 科室
        self.complications = [] # 并发症
        self.medical = []         # 药品
        self.check = []         # 检查

        self.diseaseSet = []
        self.aliasSet = []
        self.symptomsSet = []
        self.positionSet = []
        self.ConcurrenDiseasesSet = []
        self.checkSet = []
        self.departmentSet = []
        self.medicalSet = []

        # 疾病的属性
        self.diseaseInfos = []
        # 关系
        self.disease_to_symptom = []        # 疾病与症状关系
        self.disease_to_alias = []          # 疾病与别名关系
        self.disease_to_position = []       # 疾病与部位关系
        self.disease_to_department = []     # 疾病与科室关系
        self.disease_to_complication = []   # 疾病与并发症关系
        self.disease_to_check = []          # 疾病与检查
        self.disease_to_medical = []           # 疾病与药品关系   TODO

        self.readData()     # 读取数据

    def readData(self):
        """
        从数据库中读取数据
        id, diseaseName, baseInfo, aliasName, isMedicalInsurance,
        position, infectivity, MultiplePopulation, RelatedSymptoms,
        ConcurrenDiseases, department, cureCost, cureRate, curePeriod,
        check, department_one, department_two,

        :return:
        """
        sql = "select * from disease;"
        res = getQueryRes(sql)
        for item in res:
            diseaseInfo = {}
            # 疾病名称
            diseaseName = item[1]
            self.diseases.append(diseaseName)
            diseaseInfo["name"] = diseaseName

            # 基本介绍
            diseaseInfo["baseinfo"] = item[2]

            # 别名
            aliasNames = item[3]
            if aliasNames is not None:

                aliasList = re.split(self.splitStr, re.sub(r"[\"']", "", aliasNames.strip()))
                self.aliases.extend(aliasList)          # concat到最后
                self.disease_to_alias.append((diseaseName, list(set(aliasList)) ))

            # 医保
            if item[4] is not None:
                diseaseInfo["isMedicalInsurance"] = item[4]

            # 发病位置
            if item[5] is not None:
                positionList = re.split(self.splitStr, item[5].strip())
                self.position.extend(positionList)      # concat到最后
                self.disease_to_position.append((diseaseName, list(set(positionList))))

            # 传染性
            if item[6] is not None:
                diseaseInfo["infectivity"] = item[6].strip()

            # 高发人群
            if item[7] is not None:
                multiplepopList = re.split(self.splitStr, item[7].strip())
                multiplepopList = [x for x in multiplepopList if -1 == x.find("...")]
                diseaseInfo["MultiplePopulation"] = ",".join(multiplepopList)

            # 症状
            if item[8] is not None:
                symptomList = re.split(self.splitStr, item[8].strip())
                self.symptoms.extend(symptomList)       # concat
                self.disease_to_symptom.append((diseaseName, list(set(symptomList)) ))

            # 并发疾病
            if item[9] is not None:
                conDiseaseList = re.split(self.splitStr, item[9].strip())
                # self.complications.extend(conDiseaseList)   # 测试用
                if diseaseName in conDiseaseList:
                    # print("去除重复并发疾病")
                    del conDiseaseList[conDiseaseList.index(diseaseName)]
                self.disease_to_complication.append((diseaseName, list(set(conDiseaseList)) ))

            # 科室
            if item[10] is not None:
                departmentStr = item[10].strip()
                departmentList = re.split(self.splitStr, departmentStr)
                self.departments.extend(departmentList)
                self.disease_to_department.append((diseaseName, departmentList))

            # 费用
            if item[11] is not None:
                diseaseInfo["cureCost"] = item[11].strip()

            # 治愈率
            if item[12] is not None:
                diseaseInfo["cureRate"] = item[12].strip()

            # 治疗周期
            if item[13] is not None:
                diseaseInfo["curePeriod"] = item[13].strip()

            # 检查
            if item[14] is not None:
                checkList = re.split(self.splitStr, re.sub(r"[\"']", "", item[14].strip()))
                self.check.extend(checkList)
                self.disease_to_check.append((diseaseName, list(set(checkList)) ))

            # 药
            if item[15] is not None:
                medicalList = re.split(self.splitStr, re.sub(r"[\"']", "", item[15].strip()))
                self.medical.extend(medicalList)
                self.disease_to_medical.append((diseaseName, list(set(medicalList))))

            # print(diseaseInfo)
            self.diseaseInfos.append(diseaseInfo)
        # 去重
        self.diseaseSet = list(set(self.diseases))
        self.aliasSet = list(set(self.aliases))
        self.symptomsSet = list(set(self.symptoms))
        self.positionSet = list(set(self.position))
        # self.ConcurrenDiseasesSet = list(set(self.symptoms))  # 测试用， 实际上还用diseaseName
        self.checkSet = list(set(self.check))
        self.departmentSet = list(set(self.departments))
        self.medicalSet = list(set(self.medical))
        assert len(self.diseases) == len(self.diseaseInfos)
        print("Read Done")

    def createNode(self, label, nodeNames, params=None):
        """
        创建节点
        :return:
        """
        index = 1
        for nodeName in nodeNames:
            node = Node(label, name=nodeName)
            if params is not None:
                node.update(params)

            self.graph.create(node)
            print(f"[{index}]", "createNode: ", label, " ", node["name"])
            index += 1

    def createDiseaseNode(self, diseaseInfos):
        """
        创建疾病节点，因为参数全部在dict里面，直接update比较方便
        :param diseaseInfos:
        :return:
        """
        index = 1
        if diseaseInfos is not None:
            for disease in diseaseInfos:
                node = Node("Disease")
                node.update(disease)
                self.graph.create(node)
                print(f"[{index}]", "createDiseaseNode:", node["name"])
                index += 1

    def createLinks(self, start_node_ontology, end_node_ontology, edges, link_type, link_name):
        """
        :param start_node_ontology: 起点本体名称
        :param end_node_ontology:   终点本体名称
        :param edges:       边的关系
        :param link_type:   链接的类型，ALIAS_IS
        :param link_name:   链接的注释，是链接的一个属性
        :return:

        # 关系要去重！
        """
        query_raw = "match (p:%s), (q:%s) where p.name='%s' and q.name='%s' create (p)-[link:%s{name:'%s'}]->(q)"
        index = 1
        for link in edges:
            obj, subjects = link
            for subject in subjects:
                query = query_raw % (start_node_ontology, end_node_ontology, obj, subject, link_type, link_name)
                print(f"[{index}]", start_node_ontology, link_type, end_node_ontology)
                self.graph.run(query)
                index += 1

    def buildNodes(self):
        """
        建立所有实体
        :return:
        """
        self.createDiseaseNode(self.diseaseInfos)
        self.createNode("Alias", self.aliasSet)
        self.createNode("Symptom", self.symptomsSet)
        self.createNode("Position", self.positionSet)
        self.createNode("Check", self.checkSet)
        self.createNode("Department", self.departmentSet)
        self.createNode("Medical", self.medicalSet)
        # TODO 加上药物实体

    def buildLink(self):
        """
        建立链接
        self.disease_to_symptom = []        # 疾病与症状关系
        self.disease_to_alias = []          # 疾病与别名关系
        self.disease_to_position = []           # 疾病与部位关系
        self.disease_to_department = []     # 疾病与科室关系
        self.disease_to_complication = []   # 疾病与并发症关系
        self.disease_to_drug = []           # 疾病与药品关系
        self.disease_to_check = []          # 疾病与检查
        :return:
        """
        self.createLinks("Disease", "Symptom", self.disease_to_symptom, "HAS_SYMPTOM", "症状")
        self.createLinks("Disease", "Department", self.disease_to_department, "BELONG_TO_DEPARTMENT", "部门")
        self.createLinks("Disease", "Position", self.disease_to_position, "POSITION_ON", "部位")
        self.createLinks("Disease", "Check", self.disease_to_check, "NEED_CHECK", "检查")
        self.createLinks("Disease", "Alias", self.disease_to_alias, "ALIAS_AS", "别名")
        self.createLinks("Disease", "Medical", self.disease_to_medical, "NEED_MEDICAL", "用药")
        self.createLinks("Disease", "Disease", self.disease_to_complication, "HAS_CONCURRENT_DISEASE", "并发症")

    def cleanLinks(self):
        sql = "match ()-[r]->() delete r"
        self.graph.run(sql)

    def cleanNodes(self):
        sql = "match (n) delete n"
        self.graph.run(sql)


def createGraph():
    """
    此方法创建一个知识图谱
    :return:
    """
    g = QAGraph()
    # g.readData()
    g.buildNodes()
    g.buildLink()
    return g


if __name__ == "__main__":
    g = QAGraph()
    # g.buildNodes()
    g.buildLink()
    # g.cleanLinks()
