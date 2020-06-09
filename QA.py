from entityExtractor import EntityExtractor
from answerSearching import AnswerSearching
from buildGraphy import QAGraph

qagraph = QAGraph()


def get_QA_answer(question):
    # 第一步：提取实体
    entityExtractor = EntityExtractor(qagraph)
    entities = entityExtractor.extractor(question)

    print(entities)

    # # 第二步：构造sql
    ans = AnswerSearching(qagraph)
    sqls = ans.entity2SQL(entities)
    print(sqls)

    # # 第三步：查询答案
    final_answer = ans.searching(sqls)
    return final_answer


if __name__ == "__main__":
    res = get_QA_answer("哮喘属于哪个科室")
    print(res)
