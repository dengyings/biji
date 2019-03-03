import nltk.stem as ns
words = ['table', 'probably', 'wolves', 'playing', 'is',
         'dog', 'the', 'beaches', 'grounded', 'dreamt', 'envision']
lemmatizer = ns.WordNetLemmatizer()
for word in words:
    lemma = lemmatizer.lemmatize(word, 'n')
    print(lemma)
print('-' * 72)
for word in words:
    lemma = lemmatizer.lemmatize(word, 'v')
    print(lemma)
