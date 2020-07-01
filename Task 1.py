from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import  TfidfVectorizer

from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print('Initial score: ',score)



### Task 1 - a (SVM method)
svcClassifier = SVC(kernel='linear')
svcClassifier.fit(X_train_tfidf, twenty_train.target)
predict = svcClassifier.predict(X_test_tfidf)

score_svm = metrics.accuracy_score(twenty_test.target, predict)
print("Score using SVM: ", score_svm)



### Task 1 - b

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer(ngram_range=(1,2))
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score_ngram = metrics.accuracy_score(twenty_test.target, predicted)
print("Part - B ngram - Score: ",score_ngram)



# Task 1 - c
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score_english = metrics.accuracy_score(twenty_test.target, predicted)
print("Part - C stop word english score: ",score_english)


