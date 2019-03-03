import nltk.corpus as nc
doc = ' '.join(nc.brown.words()[:310])
print(doc)
words = doc.split()
print(words)
chunks = []
for word in words:
    if len(chunks) == 0 or len(chunks[-1]) == 5:
        chunks.append([])
    chunks[-1].append(word)
for chunk in chunks:
    for word in chunk:
        print('{:15}'.format(word), end=' ')
    print()
