
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from statistics import mode
from sklearn.externals import joblib

cv = joblib.load('pickles\\cv.sav')
mnb_classifier = joblib.load('pickles\\mnb_classifier.sav')
lr_classifier = joblib.load('pickles\\lr_classifier.sav')
sgd_classifier = joblib.load('pickles\\sgd_classifier.sav')
lsvc_classifier = joblib.load('pickles\\lsvc_classifier.sav')
lr_classifier = joblib.load('pickles\\lr_classifier.sav')

def find_features(text):

    tokenizer = RegexpTokenizer(r'[A-z]+')
    allowed_words = ['J', 'R', 'V', 'N']
    x_text_cv = []

    r_prc = ' '.join(w[0].lower() for w in nltk.pos_tag(word_tokenize(text)) if w[1][0] in allowed_words and len(tokenizer.tokenize(w[0].lower())) > 0) #and len(tokenizer.tokenize(w[0].lower())) > 0
    x_text_cv.append(r_prc)

    features = cv.transform(x_text_cv)

    return features

class VoteClassifier():

    def __init__(self, *classifiers):

        self.classifiers = classifiers

    def classify(self, text):
    
        votes = []

        for clf in self.classifiers:
            
            x_test_cv = find_features(text)
            pred = clf.predict(x_test_cv[0])
            vote = int(pred[0])

            if vote == 0:
                votes.append('negative')
            else:
                votes.append('positive')

        return mode(votes)

    def confidence(self, text):

        votes = []
        for clf in self.classifiers:

            x_test_cv = find_features(text)
            pred = clf.predict(x_test_cv[0])
            vote = int(pred[0])
            votes.append(vote)
        
        choicevotes = votes.count(mode(votes))
        conf_pct = (choicevotes / len(votes))

        return conf_pct*100

def sentiment(text):

    voted_classifier = VoteClassifier(mnb_classifier, lr_classifier, sgd_classifier, lsvc_classifier, lsvc_classifier)
    
    return 'Sentiment: ',voted_classifier.classify(text), 'Confidence = ',voted_classifier.confidence(text)

print(sentiment('''a decent movie'''))