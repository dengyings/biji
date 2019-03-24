import jieba

def get_content(path):
    with open(path, 'r', encoding='gbk', errors='ignore') as f:
        content = ''
        for l in f:
            l = l.strip()
            content += l
        return content

def get_TF(words, topK=10):
    tf_dic = {}
    for w in words:
        tf_dic[w] = tf_dic.get(w, 0) + 1
    return sorted(tf_dic.items(), key=lambda x: x[1], reverse=True)[:topK]

def stop_words(path):
    with open(path) as f:
        return [l.strip() for l in f]


# 分词
def main():
    path = '/home/tarena/PycharmProjects/untitled/NLP/date/机器学习.txt'
    corpus = get_content(path)
    #print(corpus)
    split_words = [x for x in jieba.cut(corpus) if x not in stop_words('/home/tarena/PycharmProjects/untitled/NLP/date/哈工大停用词表.txt')]
    print('样本分词效果：' + '/ '.join(split_words))
    print('样本的topK（10）词：' + str(get_TF(split_words)))


main()