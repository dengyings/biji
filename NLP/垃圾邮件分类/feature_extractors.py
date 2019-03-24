"""

@author: liushuchun
"""

from sklearn.feature_extraction.text import CountVectorizer


def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    #如果某个词的小于min_df，则这个词不会被当作关键词
    #ngram_range词组切分的长度范围
    features = vectorizer.fit_transform(corpus)#将原始训练和测试文本转化为特征向量
    return vectorizer, features


from sklearn.feature_extraction.text import TfidfTransformer


def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix


from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    print(vectorizer, features)
    return vectorizer, features


norm_train_corpus = ['小明　小红　今天　明天']
bow_extractor(norm_train_corpus)

