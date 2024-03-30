import json
import numpy as np

import nltk
from nltk.stem.porter import PorterStemmer

nltk.download("punkt", quiet=True)
stemmer = PorterStemmer()


def tokenize(sentence):
	return nltk.word_tokenize(sentence)


def stem(word):
	return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
	sentence_words = [stem(word) for word in tokenized_sentence]
	bag = np.zeros(len(words), dtype=np.float32)
	for idx, w in enumerate(words):
		if w in sentence_words:
			bag[idx] = 1
	return bag


def prepare_data(path):
	with open(path, "r") as f:
		intents = json.load(f)

	all_words = []
	tags = []
	xy = []
	for intent in intents["intents"]:
		tag = intent["tag"]
		tags.append(tag)
		for pattern in intent["patterns"]:
			w = tokenize(pattern)
			all_words.extend(w)
			xy.append((w, tag))

	ignore_words = ["?", ".", "!"]
	all_words = [stem(w) for w in all_words if w not in ignore_words]
	all_words = sorted(set(all_words))
	tags = sorted(set(tags))

	x_train = []
	y_train = []
	for (pattern_sentence, tag) in xy:
		bag = bag_of_words(pattern_sentence, all_words)
		x_train.append(bag)
		label = tags.index(tag)
		y_train.append(label)

	x_train = np.array(x_train)
	y_train = np.array(y_train)

	return x_train, y_train, all_words, tags, intents