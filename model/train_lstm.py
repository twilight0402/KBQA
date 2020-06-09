import torch
import jieba
import torch.nn as nn
from torch.optim import Adam
from torchtext.vocab import Vectors
from torchtext.data import BucketIterator

from sklearn.metrics import precision_recall_fscore_support
from model.Classification import Classification
from model.tool import build_and_cache_dataset

data_dir = r"E:\Workspaces\Python\KG\QA_healty39\data"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def train():
    # 获得数据集
    fields, train_dataset = build_and_cache_dataset()

    # Build vocab 获得词向量
    QUESTION, INTENTION = fields
    # 词向量是文本文件，共365113个字，第一行为介绍，第一列是词，后面的列为向量，空格分割
    vectors = Vectors(name=data_dir + r"\embedding\sgns.sogou.word", cache=data_dir + r"\embedding")
    # 建立数据集和词向量之间的对应关系
    QUESTION.build_vocab(
        train_dataset,
        max_size=40000,
        vectors=vectors,
        unk_init=torch.nn.init.xavier_normal_,
    )
    INTENTION.build_vocab(train_dataset)
    print("pad_token:", QUESTION.pad_token)
    print("stoi[QUESTION.pad_token]:", QUESTION.vocab.stoi[QUESTION.pad_token])

    model = Classification(
        vocab_size=len(QUESTION.vocab),     # 词表的长度， 8188
        output_dim=8,                       # 输出的维度， 默认5
        pad_idx=QUESTION.vocab.stoi[QUESTION.pad_token],    # 填充的id
        dropout=0.3,
    )
    model.embedding.from_pretrained(QUESTION.vocab.vectors)
    # QUESTION.vocab.stoi

    bucket_iterator = BucketIterator(
        train_dataset,
        batch_size=200,
        device=device,
    )

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(),
                     lr=0.00001,
                     eps=1e-8)
    global_step = 0
    model.zero_grad()
    for _ in range(10000):
        for step, batch in enumerate(bucket_iterator):
            model.train()
            questions, questions_lengths = batch.question
            intention = batch.intention
            # print(questions.shape)      # 7, 108
            preds = model(questions)

            loss = criterion(preds, intention)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            print(global_step, " ", loss.item())
    return model, QUESTION.vocab, INTENTION.vocab


def predict(ques_vocab, inten_vocab, model, questionList=["哮喘怎么治疗"]):
    """
    根据问题预测意图
    :param ques_vocab:  问题的词表
    :param inten_vocab:  意图的词表
    :param model:  lstm模型
    :param questionList:  问题列表（字符串列表）
    :return: intention列表（字符串列表）
    """
    sentenceList = [jieba.lcut(question) for question in questionList]
    max_len = 0
    for sentence in sentenceList:
        if len(sentence) > max_len:
            max_len = len(sentence)

    wordIDList = []
    for i in range(len(sentenceList)):
        wordList = [1] * max_len
        wordList[:len(sentenceList[i])] = [ques_vocab.stoi[word] for word in sentenceList[i]]
        wordIDList.append(wordList)

    # print("wordIDList: ", wordIDList)
    wordIDList = torch.tensor(wordIDList).transpose(0, 1).to(device)
    # print(wordIDList)
    model.eval()
    intentionid = model(wordIDList)
    # print(intentionid)
    intention = [inten_vocab.itos[intention.argmax()] for intention in intentionid]
    # print(intention)
    return intention


if __name__ == "__main__":
    from model.tool import save_Params
    from model.tool import load_Params

    path = "E:/Workspaces/Python/KG/QA_healty39/data/output"
    quesList = ["#怎么治疗", "#要怎么治疗", "#要怎么治疗", "#有什么症状", "#属于哪个科室",
                "#的治疗周期有多久", "哪些症状说明得了#", "多大概率能治好#", "#要体检哪些项目", "介绍一下#",
                "#是什么病", "介绍一下#", "得了#咋整"]
    model, QUEST_vocab, INTENT_vocab = train()
    #
    res = predict(QUEST_vocab, INTENT_vocab, model, quesList)
    # res = predict(QUEST_vocab, INTENT_vocab, model, quesList)

    save_Params(model, QUEST_vocab, INTENT_vocab, path)               # 保存
    # model_, QUEST_vocab_, INTENT_vocab_ = load_Params(path, device)     # 读取
    # res = predict(QUEST_vocab_, INTENT_vocab_, model_, quesList)
    print(res)


