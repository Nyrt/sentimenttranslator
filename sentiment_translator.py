import numpy as np 
from nltk.corpus import wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import string


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
		return "".join(c for c in word.strip() if c not in string.punctuation)

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

sentence = raw_input("Enter a sentence\n")

words = sentence.split()

sentim_toy(sentence)









# Read and format data

# Analyze sentiment

# While sentiment doesn't match desired sentiment:


	# Find the word that most strongly contributes to the sentiment

	# Find the closest antonym to that word

	# Check if sentence is still valid

	# If not, or no antonym found, and word is an adjective, try adding "not"

	# Check if sentence is still valid

	# If not, or no antonym found, and word is a verb, try adding "did not" or "do not"