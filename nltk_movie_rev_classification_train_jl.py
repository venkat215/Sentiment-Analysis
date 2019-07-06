import nltk
import random
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import movie_reviews
from sklearn.externals import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC

import pandas as pd
from statistics import mode

documents = []

# stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'[A-z]+')
allowed_words = ['J', 'R', 'V', 'N']

with open('movie_reviews_better\\positive.txt','r') as rev_file:
    rev_file_data = rev_file.read()
    for r in rev_file_data.split('\n'):
        r_prc = ' '.join(w[0].lower() for w in nltk.pos_tag(word_tokenize(r)) if w[1][0] in allowed_words and len(tokenizer.tokenize(w[0].lower())) > 0) #and len(tokenizer.tokenize(w[0].lower())) > 0
        documents.append((r_prc, 'pos'))

    rev_file.close()

with open('movie_reviews_better\\negative.txt','r') as rev_file:
    rev_file_data = rev_file.read()
    for r in rev_file_data.split('\n'):
        r_prc = ' '.join(w[0].lower() for w in nltk.pos_tag(word_tokenize(r)) if w[1][0] in allowed_words and len(tokenizer.tokenize(w[0].lower())) > 0) #and len(tokenizer.tokenize(w[0].lower())) > 0
        documents.append((r_prc, 'neg'))

    rev_file.close()

documents_large = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

for r in documents_large:

    r_prc = ' '.join(w[0].lower() for w in nltk.pos_tag(r[0]) if w[1][0] in allowed_words and len(tokenizer.tokenize(w[0].lower())) > 0) #and len(tokenizer.tokenize(w[0].lower())) > 0
    documents.append((r_prc, r[1]))

del documents_large

random.shuffle(documents)

df = pd.DataFrame(documents)

x_reviews = df[0]
y_classes = df[1]

encoder = LabelEncoder()
y_labels = encoder.fit_transform(y_classes)

cv = TfidfVectorizer(min_df=1, stop_words='english')

x_train, x_test, y_train, y_test = train_test_split(x_reviews, y_labels, test_size = 0.1, random_state = 4)
x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.transform(x_test)

joblib.dump(cv, "pickles\\cv.sav")

classifiers = [('mnb_classifier' , MultinomialNB()),
               ('sgd_classifier', SGDClassifier()),
               ('lr_classifier', LogisticRegression()),
               ('lsvc_classifier', LinearSVC()),
               ('lrcv_classifier', LogisticRegressionCV())]


for clf_name, clf in classifiers:

    clf.fit(x_train_cv, y_train)
    confidence = clf.score(x_test_cv, y_test)
    print(clf_name + '_accuracy:',confidence*100)

    joblib.dump(clf, "pickles\\" + clf_name + ".sav")


