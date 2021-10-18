from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

import numpy as np

categories = ["Breton", "Île de France"]

with open("bretagne_cut.txt") as f:
    liste_finistère = f.read().splitlines()

with open("ile_de_france_cut.txt") as f:
    liste_idf = f.read().splitlines()

target = [0 for i in range(len(liste_finistère))]
for i in liste_idf:
    target.append(1)

# twenty_train.data.lower

data = liste_finistère
for commune in liste_idf:
    data.append(commune)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data)

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, target)

docs_new = ['Brest', 'Paris', 'Loprevaler', 'Marcq', "Plouarzel", "Ploermel", "La Faloise"]
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, categories[category]))


# text_clf = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', SGDClassifier(loss='hinge', penalty='l2',
#                           alpha=1e-3, random_state=42,
#                           max_iter=5, tol=None)),
# ])

# text_clf.fit(data, target)

# for doc, category in zip(docs_new, predicted):
#     print('%r => %s' % (doc, categories[category]))
