from entityExtractor import EntityExtractor
from answerSearching import AnswerSearching
from buildGraphy import QAGraph

qagraph = QAGraph()
entityExtractor = EntityExtractor(qagraph)
ans = AnswerSearching(qagraph)


def get_QA_answer(question):
    # 第一步：提取实体
    entities = entityExtractor.extractor(question)

    # print(entities)

    # # 第二步：构造sql
    sqls = ans.entity2SQL(entities)

    # print(sqls)

    # # 第三步：查询答案
    final_answer = ans.searching(sqls)
    return final_answer


if __name__ == "__main__":
    while True:
        question = input("\n请输入你的问题：(输入Q退出)\n")
        if question == "Q" or question == "q":
            break
        res = get_QA_answer(question)
        print(res)

    print("退出")
