from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
import sklearn_crfsuite
import numpy as np


# Assignment 4: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.
def word2featuresP2(sent, i):
	word = sent[i][0]
	postag = sent[i][1]

	features = {
		'word.lower()': word.lower(),
		'word[-3:]': word[-3:],
		'word[-2:]': word[-2:],
		'word.isupper()': word.isupper(),
		'word.istitle()': word.istitle(),
		'word.isdigit()': word.isdigit(),
		'postag': postag,
		'postag[:2]': postag[:2],
	}
	if i > 0:
		word1 = sent[i - 1][0]
		postag1 = sent[i - 1][1]
		features.update({
			'-1:word.lower()': word1.lower(),
			'-1:word.istitle()': word1.istitle(),
			'-1:word.isupper()': word1.isupper(),
			'-1:postag': postag1,
			'-1:postag[:2]': postag1[:2],
		})
	else:
		features['BOS'] = True

	if i < len(sent) - 1:
		word1 = sent[i + 1][0]
		postag1 = sent[i + 1][1]
		features.update({
			'+1:word.lower()': word1.lower(),
			'+1:word.istitle()': word1.istitle(),
			'+1:word.isupper()': word1.isupper(),
			'+1:postag': postag1,
			'+1:postag[:2]': postag1[:2],
		})
	else:
		features['EOS'] = True

	return features


def sent2features(sent):
	return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
	return [label for token, postag, label in sent]


def sent2tokens(sent):
	return [token for token, postag, label in sent]


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
		(o + 'word', word.lower()), (o + 'shape', pattern)

		# TODO: add more features here.
	]
	# print(features)
	return features


def getShape(word):
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
	if word[0].isalpha():
		special = 1

	# if word[0].isupper():
	#     upper = 1

	# the window around the token
	# lmr = ""
	# mr = ""
	# lm = ""
	for o in [-2, -1, 0, 1, 2]:
		if len(sent) > i + o >= 0:
			word = sent[i + o][0]
			featlist = getfeats(word, o)
			# shape = getShape(word)
			# if o == -1:
			#     lmr += shape
			#     lm += shape
			# elif o == 0 :
			#     lmr += shape
			#     lm += shape
			#     mr += shape
			# else:
			#     lmr += shape
			#     mr += shape

			# featlist.append((str(o) + 'wordpos', sent[i + o][1]))
			features.extend(featlist)
	# features.extend([('lmr', lmr),('lm', lm), ('mr', mr)])
	# features.extend([('word', word)])
	# features.extend([('word[-3:]', word[-3:])])
	# features.extend([('word[:2]', word[2:])])
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

	X_train = [sent2features(s) for s in train_sents]
	y_train = [sent2labels(s) for s in train_sents]
	X_test = [sent2features(s) for s in test_sents]
	y_test = [sent2labels(s) for s in test_sents]


	model = sklearn_crfsuite.CRF(
		algorithm='lbfgs',
		c1=0.1,
		c2=0.1,
		max_iterations=100,
		all_possible_transitions=True
	)
	model.fit(X_train, y_train)
	# TODO: play with other models

	y_pred = model.predict(X_test)

	j = 0
	print("Writing to results.txt")
	# format is: word gold pred
	with open("results.txt", "w") as out:
		for sent in test_sents:
			for i in range(len(sent)):
				word = sent[i][0]
				gold = sent[i][-1]
				pred = y_pred[j][i]
				out.write("{}\t{}\t{}\n".format(word, gold, pred))
			j += 1
		out.write("\n")

	print("Now run: python conlleval.py results.txt")






