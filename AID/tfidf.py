import sklearn.datasets as sd
import sklearn.feature_extraction.text as ft
import sklearn.naive_bayes as nb
cld = {'misc.forsale': 'SALES',
       'rec.motorcycles': 'MOTORCYCLES',
       'rec.sport.baseball': 'BASEBALL',
       'sci.crypt': 'CRYPTOGRAPHY',
       'sci.space': 'SPACE'}
train = sd.fetch_20newsgroups(
    subset='train', categories=cld.keys(), shuffle=True,
    random_state=7)
train_data = train.data
train_y = train.target
categories = train.target_names
print(len(train_data))
print(len(train_y))
print(len(categories))
cv = ft.CountVectorizer()
train_tfmat = cv.fit_transform(train_data)
print(train_tfmat.shape)
tf = ft.TfidfTransformer()
train_x = tf.fit_transform(train_tfmat)
model = nb.MultinomialNB()
model.fit(train_x, train_y)
test_data = [
    'The curveballs of right handed pitchers tend to curve to the left',
    'Caesar cipher is an ancient form of encryption',
    'This two-wheeler is really good on slippery roads']
test_tfmat = cv.transform(test_data)
test_x = tf.transform(test_tfmat)
pred_test_y = model.predict(test_x)
for sentence, index in zip(test_data, pred_test_y):
    print(sentence, '->', cld[categories[index]])
