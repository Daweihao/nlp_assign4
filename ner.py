from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
import sklearn_crfsuite
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import ComplementNB
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# Assignment 4: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.

def getfeats(word, o):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    pattern = getShape(word)
    upper = 0
    dash = 0
    special = 0
    if word[0].isupper():
         upper = 1
    elif not word[0].isalpha():
        special = 1
    for i in range(len(word)):
            if i != 0 and i != len(word) - 1 and word[i] == '-':
                dash = 1
    features = [
        (o + 'word', word.lower()) ,(o+'shape', pattern) 

        # TODO: add more features here.
    ]
    # print(features)
    return features
    
def getShape (word):
    pattern = ""
    for i in range(len(word)):
        # if i != 0 and i != len(word) - 1 and word[i] == '-':
            # dash = 1
        if word[i].isupper():
            pattern += "A"
        elif word[i].islower():
            pattern += "a"
        elif word[i].isnumeric():
            pattern += "0"
        elif word[i] == '-' or word[i] == '.':
            pattern += "-"
    return pattern

def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    word = sent[i][0]
    pos = sent[i][1]
    special = 0
    # pattern = ""
    # upper = 0
    # dash = 0
    if  word[0].isalpha():
        special = 1

    # if word[0].isupper():
    #     upper = 1

    # the window around the token
    # lmr = ""
    # mr = ""
    # lm = ""
    for o in [-2,-1,0,1,2]:
        if len(sent) > i + o >= 0:
            word = sent[i + o][0]
            featlist = getfeats(word, o)
            #featlist.append((str(o) + 'wordpos', sent[i + o][1]))
            features.extend(featlist)
    # features.extend([('word', word)])
    # features.extend([('word[-3:]', word[-3:])])
     #features.extend([('word[:2]', word[2:])])
    # features.extend([('length', len(word))])
    features.extend([('pos', pos)])
    # features.extend([('Pattern', pattern)])
    # features.extend([('special', special)])
    # features.extend([('upper', upper)])
    # features.extend([('dash', dash)])
    print(features)

    return dict(features)

if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

    # train_sents.extend(dev_sents)
    train_feats = []
    train_labels = []



    for sent in train_sents:
        for i in range(len(sent)):
            feats = word2features(sent, i)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    # model = SGDClassifier()
    # classes = len(np.unique(train_labels))
    # classes = classes.tolist()
    # model = PassiveAggressiveClassifier()
    model = Perceptron(verbose=1)
    # model = MultinomialNB()
    # model = RandomForestClassifier()
    # model = DecisionTreeClassifier()
    # model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3))
    # model = LogisticRegression()
    model.fit(X_train, train_labels)
    # TODO: play with other models


    test_feats = []
    test_labels = []

    # switch to test_sents for your final results
    for sent in test_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)

    j = 0
    print("Writing to results.txt")
    # format is: word gold pred
    with open("results.txt", "w") as out:
        for sent in test_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
                j += 1
        out.write("\n")

    print("Now run: python conlleval.py results.txt")






