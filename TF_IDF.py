import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer(analyzer="char", lowercase=False)


def cutall(sentence):
    return list(sentence)


def getCutRes(questions):
    questions_ = []
    for sentence in questions:
        sentence = " ".join(cutall(sentence))
        questions_.append(sentence)
    return questions_


def getTFIDF(questions):
    corpus = getCutRes(questions)
    tfidf_vec.fit(corpus)
    tfidf_matrix = tfidf_vec.transform(corpus)
    return tfidf_matrix.todense(), tfidf_vec


def getFeatures(questions):
    tfidf = getTFIDF(questions)
    return tfidf


if __name__ == "__main__":
    res = getTFIDF(["哮喘病怎么治", "心脏病怎么治"])
    print(res)
