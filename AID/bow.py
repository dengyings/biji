import nltk.tokenize as tk
import sklearn.feature_extraction.text as ft
doc = 'The brown dog is running. The black dog is in the black room. Running in the room is forbidden.'
print(doc)
print('-' * 72)
sentences = tk.sent_tokenize(doc)
print(sentences)
print('-' * 72)
cv = ft.CountVectorizer()
tfmat = cv.fit_transform(sentences).toarray()
words = cv.get_feature_names()
print(words)
print(tfmat)
