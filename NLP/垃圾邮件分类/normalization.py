"""

@author: liushuchun
"""
import re
import string
import jieba

# 加载停用词
with open("dict/stop_words.utf8", encoding="utf8") as f:
    stopword_list = f.readlines()


def tokenize_text(text):
    tokens = jieba.cut(text)
    tokens = [token.strip() for token in tokens]
    return tokens

def remove_special_characters(text):
    tokens = tokenize_text(text)#分词，去空格
    pattern = re.compile('[{}，。]'.format(re.escape(string.punctuation)))#string.punctuation #所有的标点字符
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])#第一个参数为None，则直接将第二个参数中为True的值筛选出来。
    filtered_text = ' '.join(filtered_tokens)#拼接
    return filtered_text

def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ''.join(filtered_tokens)
    return filtered_text


def normalize_corpus(corpus, tokenize=False):
    normalized_corpus = []
    for text in corpus:
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        normalized_corpus.append(text)
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
    return normalized_corpus

