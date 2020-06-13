from buildGraphy import QAGraph


class AnswerSearching:
    def __init__(self, graph: QAGraph):
        self.qagraph = graph

    def searching(self, sqls):
        if sqls is None:
            return None
        intention = sqls["intention"]
        if sqls.get("sql") is None:
            return "这题我不会啊， 请换一种方式提问"

        sqlList = sqls["sql"]

        res = {"intention": intention, "result": []}
        for sql in sqlList:
            temp = self.qagraph.graph.run(sql).data()
            res["result"].append(temp)

        answer = self.parseQuery2NL(res)
        return answer

    def parsedict(self, data, *args):
        """
        args: d.name, s.name
        :param data:
        :param args:
        :return:
        """
        if len(args) == 1:
            res = []
            for item in data:
                res.append(item[args[0]])
            return res
        elif len(args) == 2:
            res_dict = {}
            for item in data:
                if res_dict.get(item[args[0]]) is None:
                    res_dict[item[args[0]]] = [item[args[1]]]
                else:
                    res_dict[item[args[0]]].append(item[args[1]])
            return list(res_dict.items())[0]

    def parseQuery2NL(self, data):
        """
        将查询结果转换为可读的字符串
        1.查询症状 query_symptom(disease, symptom)                      \n
        2.查询治疗方法 query_cureway(disease, medical)                   \n
        3.查询治疗周期 query_period (disease, period)                    \n
        4.查询治愈率 query_rate(disease, period)                         \n
        5.查询检查项目 query_checklist (disease, ckeck)                   \n
        6.查询科室 query_department (disease, department)                \n
        7.查询疾病(根据其他属性查询疾病) query_disease (disease)             \n
        8.查询疾病描述 disease_describe(disease, baseinfo)				   \n

        :param data:
        :return:
        """
        intention = data["intention"]
        result = data["result"][0]     # [dict]

        assert intention is not None

        if intention == "query_symptom":
            res_dict = self.parsedict(result, "d.name", "s.name")
            answer = f"{res_dict[0]}的症状有：{','.join(res_dict[1])}"

        elif intention == "query_cureway":
            res_tuple = self.parsedict(result, "d.name", "n.name")
            answer = f"得了{res_tuple[0]}可服用：{','.join(res_tuple[1])}"

        elif intention == "query_period":
            res_tuple = self.parsedict(result, "d.name", "d.curePeriod")
            answer = f"{res_tuple[0]}的治疗周期为：{','.join(res_tuple[1])}"

        elif intention == "query_rate":
            res_tuple = self.parsedict(result, "d.name", "d.cureRate")
            answer = f"{res_tuple[0]}的治愈率为：{','.join(res_tuple[1])}"

        elif intention == "query_checklist":
            res_tuple = self.parsedict(result, "d.name", "c.name")
            answer = f"{res_tuple[0]}需要做以下检查：{','.join(res_tuple[1])}"

        elif intention == "query_department":
            res_tuple = self.parsedict(result, "d.name", "n.name")
            departmentList = res_tuple[1]
            if len(departmentList) == 1:
                answer = f"{res_tuple[0]}可以挂 {departmentList[0]}"
            else:
                depart_desc = "、".join(departmentList[:-1])
                depart_desc += "和" + departmentList[-1]
                answer = f"{res_tuple[0]}可以去{depart_desc}"

        elif intention == "query_disease":
            res_list = self.parsedict(result, "d.name")
            answer = f"可能的疾病有：{','.join(res_list)}"

        elif intention == "disease_describe":
            res_tuple = self.parsedict(result, "d.name", "d.baseinfo")
            answer = f"{res_tuple[1]}"

        return answer

    def entity2SQL(self, data):
        """
        根据从问题中提取的实体，构造对应的sql \n
        intentions = ["query_symptom", "query_cureway", "query_period", "query_rate",
              "query_checklist", "query_department", "query_disease", "disease_describe"]\n

        # 实体\n
        diseases = []      # 疾病名称 \n
        aliases = []       # 别名\n
        symptoms = []      # 症状\n
        position = []      # 部位\n
        departments = []   # 科室\n
        complications = [] # 并发症\n
        medical = []         # 药品\n
        check = []         # 检查\n

        :param entities: 从question中提取的实体
        :return: {intention: "", sql: [""]}
        """
        if data is None:
            return sqls

        intent = data["intention"][0]       # 预留，intention是数组，方便后期扩展
        sql_map = {}
        sql_map["intention"] = intent
        sql = []
        if data.get("Disease"):
            sql = self.transfor_to_sql("Disease", data["Disease"], intent)
        elif data.get("Alias"):
            sql = self.transfor_to_sql("Alias", data["Alias"], intent)
        elif data.get("Symptom"):
            sql = self.transfor_to_sql("Symptom", data["Symptom"], intent)
        elif data.get("Complication"):
            sql = self.transfor_to_sql("Complication", data["Complication"], intent)
        elif data.get("Position"):
            sql = self.transfor_to_sql("Complication", data["Complication"], intent)
        elif data.get("Check"):
            sql = self.transfor_to_sql("Complication", data["Complication"], intent)
        elif data.get("Medical"):
            sql = self.transfor_to_sql("Medical", data["Medical"], intent)

        if sql:
            sql_map['sql'] = sql
        return sql_map

    def transfor_to_sql(self, label, entities, intent):
        """
        将问题转变为cypher查询语句
        :param label:实体标签
        :param entities:实体列表
        :param intent:查询意图
        :return:cypher查询语句
        ["query_symptom", "query_cureway", "query_period", "query_rate",
              "query_checklist", "query_department", "query_disease", "disease_describe"]
        """
        if not entities:
            return []
        sql = []

        # 1.查询症状 query_symptom(disease, symptom)
        ## (疾病)
        if intent == "query_symptom" and label == "Disease":
            sql = ["MATCH (d:Disease)-[:HAS_SYMPTOM]->(s) WHERE d.name='{0}' RETURN d.name,s.name".format(e)
                   for e in entities]
        ## (别名)
        if intent == "query_symptom" and label == "Alias":
            sql = ["MATCH (a:Alias)<-[:ALIAS_IS]-(d:Disease)-[:HAS_SYMPTOM]->(s) WHERE a.name='{0}' return " \
                   "d.name,s.name".format(e) for e in entities]

        # 2.查询治疗方法 query_cureway(disease, medical)
        ## (疾病名称)（用药）
        if intent == "query_cureway" and label == "Disease":
            sql = ["MATCH (d:Disease)-[:NEED_MEDICAL]->(n) WHERE d.name='{0}' return d.name, n.name".format(e) for e in entities]
        if intent == "query_cureway" and label == "Alias":
            sql = ["MATCH (n)<-[:NEED_MEDICAL]-(d:Disease)-[]->(a:Alias) " \
                   "WHERE a.name='{0}' return d.name, n.name".format(e) for e in entities]
        ## (症状)（头疼怎么办）
        if intent == "query_cureway" and label == "Symptom":
            sql = ["MATCH (n)<-[:HAS_DRUG]-(d:Disease)-[]->(s:Symptom) WHERE s.name='{0}' " \
                   "return d.name, n.name".format(e) for e in entities]

        # 3.查询治疗周期 query_period (disease, period)
        if intent == "query_period" and label == "Disease":
            sql = ["MATCH (d:Disease) WHERE d.name='{0}' return d.name,d.curePeriod".format(e) for e in entities]
        if intent == "query_period" and label == "Alias":
            sql = ["MATCH (d:Disease)-[:ALIAS_AS]->(a:Alias) WHERE a.name='{0}' return d.name,d.curePeriod".format(e)
                   for e in entities]
        if intent == "query_period" and label == "Symptom":
            sql = ["MATCH (d:Disease)-[]->(s:Symptom) WHERE s.name='{0}' return d.name,d.curePeriod".format(e)
                   for e in entities]

        # 4.查询治愈率 query_rate(disease, period)
        # （疾病）
        if intent == "query_rate" and label == "Disease":
            sql = ["MATCH (d:Disease) WHERE d.name='{0}' return d.name,d.cureRate".format(e) for e in entities]
        # （别名）
        if intent == "query_rate" and label == "Alias":
            sql = ["MATCH (d:Disease)-[]->(a:Alias) WHERE a.name='{0}' return d.name,d.cureRate".format(e)
                   for e in entities]
        # （症状）
        if intent == "query_rate" and label == "Symptom":
            sql = ["MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) " \
                   "WHERE s.name='{0}' return d.name,d.cureRate".format(e)
                   for e in entities]

        # 5.查询检查项目 query_checklist (disease, ckeck)
        # (疾病、别名)
        if intent == "query_checklist" and label == "Disease":
            sql = ["MATCH (d:Disease) -[:NEED_CHECK]->(c:Check) " \
                   "WHERE d.name='{0}' return d.name, c.name".format(e) for e in entities]
        if intent == "query_checklist" and label == "Alias":
            sql = ["MATCH (c:Check)<-[:NEED_CHECK]-(d:Disease)-[:ALIAS_AS]->(a:Alias) " \
                   "WHERE a.name='{0}' return d.name,c.name".format(e)
                   for e in entities]
        # （症状）
        if intent == "query_checklist" and label == "Symptom":
            sql = ["MATCH (s:Symptom)<-[:HAS_SYMPTOM]-(d:Disease)-[:NEED_CHECK]->(c:Check) " \
                   "WHERE s.name='{0}' return d.name, c.name".format(e) for e in entities]

        # 6.查询科室 query_department (disease, department)
        if intent == "query_department" and label == "Disease":
            sql = ["MATCH (d:Disease)-[:BELONG_TO_DEPARTMENT]->(n:Department) WHERE d.name='{0}' return d.name," \
                   "n.name".format(e) for e in entities]
        if intent == "query_department" and label == "Alias":
            sql = ["MATCH (n:Department)<-[:BELONG_TO_DEPARTMENT]-(d:Disease)-[:ALIAS_IS]->(a:Alias) " \
                   "WHERE a.name='{0}' return d.name,n.name".format(e) for e in entities]
        if intent == "query_department" and label == "Symptom":
            sql = ["MATCH (n:Department)<-[:BELONG_TO_DEPARTMENT]-(d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) " \
                   "WHERE s.name='{0}' return d.name,n.name".format(e) for e in entities]

        # 7.查询疾病(根据其他属性查询疾病) query_disease (disease)
        if intent == "query_disease" and label == "Alias":
            sql = ["MATCH (d:Disease)-[:ALIAS_AS]->(s:Alias) WHERE s.name='{0}' " \
                   "return d.name".format(e) for e in entities]
        if intent == "query_disease" and label == "Symptom":
            sql = ["MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) WHERE s.name='{0}' " \
                   "return d.name".format(e) for e in entities]

        # 8.查询疾病描述 disease_describe(disease, baseinfo)
        if intent == "disease_describe" and label == "Alias":
            sql = ["MATCH (d:Disease)-[:ALIAS_AS]->(a:Alias) WHERE a.name='{0}' " \
                   "return d.name, d.baseinfo".format(e) for e in entities]
        if intent == "disease_describe" and label == "Disease":
            sql = ["MATCH (d:Disease) WHERE d.name='{0}' return d.name, d.baseinfo".format(e) for e in entities]
        # if intent == "disease_describe" and label == "Symptom":
        #     sql = ["MATCH (d:Disease)-[]->(s:Symptom) WHERE s.name='{0}' return d.name,d.age," \
        #            "d.insurance,d.infection,d.checklist,d.period,d.rate,d.money".format(e) for e in entities]

        return sql


if __name__ == "__main__":
    pass