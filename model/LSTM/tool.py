import os
import jieba
import torch
import pickle

from torchtext.vocab import Vectors

from model.LSTM.Classification_LSTM import Classification_LSTM
from torchtext.data import Field, LabelField, TabularDataset


def build_and_cache_dataset(data_path=r"E:\Workspaces\Python\KG\QA_healty39\data"):
    """
    返回每个属性的Field，以及所有的属性的值
    (id, category, news), datasets
    (Field, Field, Field), TabularDataset
    """
    QUESTION = Field(sequential=True, tokenize=jieba.lcut, include_lengths=True)
    INTENTION = LabelField(sequential=False, use_vocab=True, is_target=True)
    fields = [
        ('question', QUESTION),
        ('intention',  INTENTION),
    ]

    # `\t` 分割
    dataset = TabularDataset(
        os.path.join(data_path, 'qa.csv'),
        format='csv',
        fields=fields,
        csv_reader_params={'delimiter': '\t'},
    )
    features = ((QUESTION, INTENTION), dataset)
    return features


## 判断头部第一个字相同 todo
def build_symptom_vector():
    """
    返回每个属性的Field，以及所有的属性的值
    (id, category, news), datasets
    (Field, Field, Field), TabularDataset
    """

    def cutall(input):
        return list(input)

    SYMPTOM = Field(sequential=True, tokenize=cutall, include_lengths=True)
    fields = [
        ('symptom',  SYMPTOM),
    ]

    # `\t` 分割
    dataset = TabularDataset(
        os.path.join(r"E:\Workspaces\Python\KG\QA_healty39\data", 'symptom.csv'),
        format='csv',
        fields=fields,
        csv_reader_params={'delimiter': '\t'},
    )
    features = (SYMPTOM, dataset)
    return features


def build_disease_vector():
    """
    返回每个属性的Field，以及所有的属性的值
    (id, category, news), datasets
    (Field, Field, Field), TabularDataset
    """

    def cutall(input):
        return list(input)

    DISEASE = Field(sequential=True, tokenize=cutall, include_lengths=True)
    fields = [
        ('disease',  DISEASE),
    ]

    # `\t` 分割
    dataset = TabularDataset(
        os.path.join(r"E:\Workspaces\Python\KG\QA_healty39\data", 'disease.csv'),
        format='csv',
        fields=fields,
        csv_reader_params={'delimiter': '\t'},
    )
    features = (DISEASE, dataset)
    return features


def load_symptom_vector():
    SYMPTOM, dataset = build_symptom_vector()
    data_dir = r"E:\Workspaces\Python\KG\QA_healty39\data"
    vectors = Vectors(name=data_dir + r"\embedding\sgns.sogou.word", cache=data_dir + r"\embedding")
    SYMPTOM.build_vocab(
        dataset,
        max_size=40000,
        vectors=vectors,
        unk_init=torch.nn.init.xavier_normal_,
    )
    return SYMPTOM.vocab


def load_disease_vector():
    DISEASE, dataset = build_disease_vector()
    data_dir = r"E:\Workspaces\Python\KG\QA_healty39\data"
    vectors = Vectors(name=data_dir + r"\embedding\sgns.sogou.word", cache=data_dir + r"\embedding")
    DISEASE.build_vocab(
        dataset,
        max_size=40000,
        vectors=vectors,
        unk_init=torch.nn.init.xavier_normal_,
    )
    return DISEASE.vocab


def saveFile(path, obj):
    with open(path, "wb") as file:
        pickle.dump(obj, file)
        # print("dump ", path)


def loadFile(path):
    with open(path, "rb") as file:
        obj = pickle.load(file)
        # print("load ", path)
    return obj


def save_Params(model, QUEST_vocab, INTENT_vocab, PATH: str):
    """
    保存模型和参数
    :param model:
    :param QUEST_vocab:
    :param INTENT_vocab:
    :param PATH:
    :return:
    """
    PATH_raw = PATH + "/{}"
    torch.save(model.state_dict(), PATH_raw.format("model.pkl"))
    # print(PATH_raw.format("model.pkl"))

    saveFile(PATH_raw.format("QUEST_vocab.pkl"), QUEST_vocab)
    saveFile(PATH_raw.format("INTENT_vocab.pkl"), INTENT_vocab)


def load_Params(PATH, device):
    """
    加载模型和参数
    :param PATH:
    :return: model, QUEST_vocab, INTENT_vocab
    """
    PATH_RAW = PATH + "/{}"
    QUEST_vocab = loadFile(PATH_RAW.format("QUEST_vocab.pkl"))
    INTENT_vocab = loadFile(PATH_RAW.format("INTENT_vocab.pkl"))

    model = Classification_LSTM(
        vocab_size=len(QUEST_vocab),        # 词表的长度， 8188
        output_dim=8,                       # 输出的维度， 默认5
        pad_idx=QUEST_vocab.stoi["<pad>"],  # 填充的id
        dropout=0.5,
    )
    model.to(device)
    model.load_state_dict(torch.load(PATH_RAW.format("model.pkl")))
    return model, QUEST_vocab, INTENT_vocab


if __name__ == "__main__":
    load_symptom_vector()

