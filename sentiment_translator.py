import numpy as np 
from nltk.corpus import wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.parse.stanford import StanfordParser
from nltk.tag.stanford import StanfordPOSTagger
import string
import os

os.environ["CLASSPATH"] = "1"

# NLTK sentiment analysis
# http://www.nltk.org/howto/sentiment.html

# Word similarity and antonym search
# http://www.nltk.org/howto/wordnet.html

# Stanford parser for valid sentence construction?
# https://github.com/nltk/nltk/wiki/Installing-Third-Party-Software 

# NLTK Parsing
# http://www.nltk.org/book/ch08.html

conversion_thresh = 0.5

def strip(word):
		return "".join(c for c in word.strip() if c not in string.punctuation and c in string.printable)

# Toy antonym testing
def antonym_toy(words):

	for i in xrange(len(words)):
		word = strip(words[i])
		anti_found = False
		for w in wn.synsets(word):
			for j in w.lemmas():
				if j.antonyms():
					words[i] = j.antonyms()[0].name() + '`'
					anti_found = True
					break
			if anti_found:
				break
		if not anti_found: #and is_adjective
			words[i] = "not " + words[i]
	print " ".join(words)

# toy sentiment analysis
def sentim_toy(sentence):
	sid = SentimentIntensityAnalyzer()
	ss = sid.polarity_scores(sentence)
	if ss['neu'] > conversion_thresh:
		print "Neutral", ss['neu']
	elif ss['pos'] > ss['neg']:
		print "positive", ss['pos']
	else:
		print "negative", ss['neg']
	#for k in sorted(ss):
		#print('{0}: {1}, '.format(k, ss[k]))
	#print()

pos_tag_model = 'stanford/stanford-postagger-2017-06-09/models/english-bidirectional-distsim.tagger'
pos_tag_jar = 'stanford/stanford-postagger-2017-06-09/stanford-postagger-3.8.0.jar'
tagger = StanfordPOSTagger(pos_tag_model, pos_tag_jar)


#Tag words and split at conjunctions
def parse_tag(words):
	clauses = []
	tags = tagger.tag(words)
	clause_start = 0
	clause_end = 0
	for word, tag in tags:
		clause_end += 1
		if tag == 'CC' or tag == 'IN':
			clauses.append(tags[clause_start:clause_end])
			clause_start = clause_end
	if clause_start != clause_end:
		clauses.append(tags[clause_start:])
	return clauses

	# parser = StanfordParser()
	# parse_str = parser.parse(sentence)
	# print parse_str
	# clauses = []
	# for subtree in Tree.fromstring(parse_str).subtrees():
	# 	if subtree.label() == "S" or subtree.label() == "SBAR":
	# 		clauses.append(''.join(subtree.leaves()))
	# clauses_bck = clauses[:]
	# print clauses
	# for i in reversed(range(len(clauses) - 1)):
	# 	clauses[i] = clauses[i][0:clauses[i].index(clauses[i+1])]
	# clauses.append(clauses_bck[0][clauses[0].index(clauses[1]) + len(clauses[1]):])
	# for clause in clauses:
	# 	print clauses


def wn_tag(treebank_tag):
	if treebank_tag.startswith('J'):
		return wn.ADJ
	elif treebank_tag.startswith('V'):
		return wn.VERB
	elif treebank_tag.startswith('N'):
		return wn.NOUN
	elif treebank_tag.startswith('R'):
		return wn.ADV
	else:
		return ''

def Translate(words):
	#print words
	for clause in words:
		new_clause = [word[0] for word in clause]
		for i in xrange(len(clause)):
			word = strip(clause[i][0])
			tag = clause[i][1]
			s_tag = wn_tag(tag)

			# Remove inverters
			# FIX STRING COMPARISON-> UNICODE STRINGS
			if word == "not":
				new_clause[i] = ""
				continue

			if word == "dont":
				new_clause[i] = "do"
				continue

			if word == "cant":
				new_clause[i] = "can"
				continue

			if word == "wont":
				new_clause[i] = "will"
				continue

			# We can't have any differing sentiments!
			if word == "but":
				new_clause[i] = "and"
				continue
			
			# Replace adjectives with antonyms
			if word in stopwords.words():
				continue

			anti_found = False
			for w in wn.synsets(word, pos=s_tag):
				for j in w.lemmas():
					if j.antonyms():

						#Conjugate words???
						new_clause[i] = " ".join(j.antonyms()[0].name().split("_")) + '`'
						anti_found = True
						break
				if anti_found:
					break
			
			# If the word has no antonym, invert it.
			if not anti_found and (s_tag == wn.ADJ or s_tag == wn.ADV):
				new_clause[i] = "not " + word
			# Invert verbs (make tense sensitive)
			elif not anti_found and s_tag == wn.VERB:
				new_clause[i] = "didn't " + word

			# Check if the sentiment has gone in the right direction

			# If not, revert the change.

		print " ".join(new_clause)


sentence = raw_input("Enter a sentence\n")

words = sentence.split()

tags = parse_tag(words)

Translate(tags)

#sentim_toy(sentence)







# Read and format data

# Analyze sentiment

# While sentiment doesn't match desired sentiment:


	# Find the word that most strongly contributes to the sentiment

	# Find the closest antonym to that word

	# Check if sentence is still valid

	# If not, or no antonym found, and word is an adjective, try adding "not"

	# Check if sentence is still valid

	# If not, or no antonym found, and word is a verb, try adding "did not" or "do not"