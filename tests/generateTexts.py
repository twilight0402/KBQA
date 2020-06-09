from utils import DB
import random

# intentions = ["query_symptom", "query_cureway", "query_period", "query_rate",
#               "query_checklist", "query_department", "query_disease", "disease_describe", "QA_matching"]
intentions = ["query_symptom", "query_cureway", "query_period", "query_rate",
              "query_checklist", "query_department", "query_disease", "disease_describe"]

questions_raw = [["%s有什么症状？", "%s的症状有哪些？", "%s有什么表现？", "得了%s会怎么样？",
             "得了%s有哪些表现？", "哪些症状说明得了%s？", "哪些现象说明得了%s？"],
             ["%s怎么治", "%s吃什么药", "%s怎么医治", "%s怎么治疗", "%s怎么医", "%s有什么治疗方法",
              "得了%s怎么办", "得了%s咋办", "得了%s咋治", "得了%s咋治疗", "得了%s怎么治", "得了%s吃什么药",
              "得了%s吃啥药", "得了%s用什么药", "%s的治疗方式", "%s的处理方式"],
             ["治疗%s需要多长时间", "治疗%s需要多久", "治好%s需要多长时间", "治好%s要多久", "治好%s要多长时间", "%s痊愈要多久", "%s痊愈要多长时间",
              "%s康复要多久", "%s康复要多长时间", "%s治好要多久", "%s治好要多长时间", "%s治好需要多长时间", "多少天能治好%s", "多久能治好%s",
              "%s的治疗周期是多长", "%s的治疗周期有多长"],
             ["%s多大概率能治好", "多大概率能治好%s", "%s多大几率能治好", "多大几率能治好%s", "治好%s的概率是多少", "治好%s的几率是多少",
              "%s有多大概率能治好", "有多大概率能治好%s", "%s有多大几率能治好", "有多大几率能治好%s", "%s治好的希望大吗",
              "%s治好的可能性大吗", "%s治好的几率大吗", "%s治好的概率大吗", "%s能治好吗", "%s能治愈吗", "能治愈%s吗",
              "%s能治好吗", "能治好%s吗", "%s能治好吗", "能治好%s吗", "%s能好吗", "%s可治吗"],
             ["%s需要检查什么", "%s要检查什么", "%s需要检查什么项目", "%s要检查什么项目", "%s要做什么检查", "%s要检查哪些项目",
              "%s要做哪些检查", "%s怎么检查", "%s要体检哪些项目", "%s要化验吗", "%s要化验哪些项目"],
             ["%s要挂哪个科", "%s要挂哪个科室", "%s是什么科", "%s是哪个科", "%s挂什么", "%s挂什么科", "%s挂哪个", "%s属于什么科",
              "%s属于什么科室", "%s属于哪个科室", "%s属于哪些科室", "%s属于哪个科", "%s属于哪些科"],
             ["%s是什么病", "%s可能是什么病", "什么病会导致%s", "什么病导致%s", "%s可能是由什么引起的", "%s是咋回事", "%s是怎么了",
              "%s是什么问题", "%s是什么情况", "%s是什么毛病", "%s是啥毛病", "%s是哪种病导致的", "哪种病会导致%s", "%s是怎么回事"],
             ["%s是什么", "什么是%s", "介绍一下%s", "%s的描述", "%s的介绍", "%s的简介", "%s是什么病", "描述一下%s"]]


def generate():
    conn = DB.getConnection()
    cursor = conn.cursor()
    sql = "SELECT diseaseName FROM disease"
    cursor.execute(sql)
    diseaseNameList = cursor.fetchall()

    x = []
    y = []

    for intention_index in range(len(intentions)):
        questions = questions_raw[intention_index]
        for count in range(10):
            for question in questions:
                random_index = random.randint(0, len(diseaseNameList))
                print(question)
                x.append(question % (diseaseNameList[random_index][0]))
                y.append(intention_index)
    return x, y


if __name__ == "__main__":
    x, y = generate()
    print("aaa")