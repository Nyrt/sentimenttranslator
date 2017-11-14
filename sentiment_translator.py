import numpy as np 
from nltk.corpus import wordnet as wn
from nltk.sentiment import SentimentAnalyzer


# http://www.nltk.org/howto/sentiment.html
# http://www.nltk.org/howto/wordnet.html

# Stanford parser for valid sentence construction
# https://github.com/nltk/nltk/wiki/Installing-Third-Party-Software 

# Toy antonym testing
sentence = raw_input("Enter a sentence\n")

words = sentence.split()

anti = ""

for i in xrange(len(words)):
	anti_found = False
	for w in wn.synsets(word):
		for j in w.lemmas():
			if j.antonyms():
				anti += j.antonyms()[0].name()+ " "
				anti_found = True
				break
		if anti_found:
			break
	if not anti_found:
		anti += word + " "
print anti


# Read and format data

# Analyze sentiment

# While sentiment doesn't match desired sentiment:


	# Find the word that most strongly contributes to the sentiment

	# Find the closest antonym to that word