import torch
import jieba
import torch.nn as nn
from torch.optim import Adam
from torchtext.vocab import Vectors
from torchtext.data import BucketIterator
import numpy as np

from model.LSTM.Classification_LSTM import Classification_LSTM
from model.LSTM.tool import build_and_cache_dataset

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

    model = Classification_LSTM(
        vocab_size=len(QUESTION.vocab),     # 词表的长度，
        output_dim=8,                       # 输出的维度， 默认8
        pad_idx=QUESTION.vocab.stoi[QUESTION.pad_token],    # 填充的id
        dropout=0.5,
    )
    model.embedding.from_pretrained(QUESTION.vocab.vectors)

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
    for _ in range(2500):
        for step, batch in enumerate(bucket_iterator):
            model.train()
            questions, questions_lengths = batch.question
            intention = batch.intention
            preds = model(questions)

            loss = criterion(preds, intention)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            a = preds.detach().cpu().numpy()
            b = intention.detach().cpu().numpy()
            a = a.argmax(axis=1)
            acc = (a == b)
            acc = np.array(acc, dtype=float)
            acc = acc.sum() / acc.shape[0]
            # print(f"acc: {acc}")
        global_step += 1
        if global_step % 100 == 0:
            print(global_step, " ", loss.item(), "acc:", acc)

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
    from model.LSTM.tool import save_Params
    from model.LSTM.tool import load_Params

    path = "E:/Workspaces/Python/KG/QA_healty39/data"
    quesList = ["#有什么表现", "哪些症状表示得了#", "有哪些现象表明得了#", "#怎么搞才行", "#怎么治才好",
           "#要怎么治疗", "#怎么医治", "得了#咋整", "治疗#需要多久", "#康复要多长时间",
           "#的治疗周期有多长", "#多大概率能治好", "#有多大概率能治好", "#治好的概率大吗", "#要做什么检查",
           "得了#需要检查什么", "得了#要做哪些检查", "#应该挂哪个科室", "得了#要挂哪个科室", "得了#应该挂哪些科",
           "@大概率是什么情况", "@是什么情况", "@是怎么回事", "#是什么疾病", "#的简介",
           "描述一下#", "#会死吗"]
    quesList = ["#会死吗"]
    # model, QUEST_vocab, INTENT_vocab = train()
    # res = predict(QUEST_vocab, INTENT_vocab, model, quesList)
    # save_Params(model, QUEST_vocab, INTENT_vocab, path)               # 保存
    #
    model_, QUEST_vocab_, INTENT_vocab_ = load_Params(path, device)     # 读取
    res = predict(QUEST_vocab_, INTENT_vocab_, model_, quesList)
    print(res)

