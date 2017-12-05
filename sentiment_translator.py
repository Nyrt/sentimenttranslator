import numpy as np 
from nltk.corpus import wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.parse.stanford import StanfordParser
from nltk.tag.stanford import StanfordPOSTagger
import string
import os
import re
from nltk.corpus import movie_reviews
from nltk.util import unique_list
from flask import Flask, render_template, request
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('ui.html')

@app.route('/my-link/')
def my_link():
    print 'I got clicked!'
    return 'Click.'





pos_tag_model = 'stanford/stanford-postagger-2017-06-09/models/english-bidirectional-distsim.tagger'
pos_tag_jar = 'stanford/stanford-postagger-2017-06-09/stanford-postagger-3.8.0.jar'
tagger = StanfordPOSTagger(pos_tag_model, pos_tag_jar)

parse_model = 'stanford/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar'
parse_jar = 'stanford/stanford-parser-full-2017-06-09/stanford-parser.jar'
parser = StanfordParser(parse_model, parse_jar)


# NLTK sentiment analysis
# http://www.nltk.org/howto/sentiment.html

# Word similarity and antonym search
# http://www.nltk.org/howto/wordnet.html

# Stanford parser for valid sentence construction?
# https://github.com/nltk/nltk/wiki/Installing-Third-Party-Software 

# NLTK Parsing
# http://www.nltk.org/book/ch08.html

conversion_thresh = 0.5

# def train_model():
#   print "preprocessing data"
#   corpus = movie_reviews.sents()
#   corpus = [parse_tag(" ".join(sent))[0] for sent in corpus]
#   tags = unique_list(tag for sent in corpus for (word, tag) in sent)
#   vocab = unique_list(word for sent in corpus for (word, tag) in sent)
#   print len(tags)
#   print len(vocab)
#   trainer = nltk.tag.HiddenMarkovModelTrainer(tag_set, vocab)
#   train_set = []
#   test_set = []
#   for i in xrange(len(corpus)):
#       if i %10 == 0:
#           test_set += [corpus[i]]
#       else:
#           train_set += [corpus[i]]
#   print len(train_set)
#   print len(test_set)


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




#Tag words and split at conjunctions
def parse_tag(text):
    sentences = text.split(".?!")
    clauses = [tagger.tag(sentence.split()) for sentence in sentences]
    # tags = tagger.tag(words)
    # clause_start = 0
    # clause_end = 0
    # for word, tag in tags:
    #   clause_end += 1
    #   if tag == 'CC' or tag == 'IN' or word[-1] in string.punctuation:
    #       clauses.append(tags[clause_start:clause_end])
    #       clause_start = clause_end
    # if clause_start != clause_end:
    #   clauses.append(tags[clause_start:])
    #print clauses
    return clauses

def is_valid(sentence):
    parse_tree = parser.parse(sentence)
    print parse_tree
    return True


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

# Returns sentiment in a range from -1 to 1 (negative to positive respectively)
def sentiment(sentence):
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(sentence)
    return (ss['pos'] - ss['neg']) * (1.0-ss['neu'])

def invert_stopwords(word):
    print word
    if word == "not":
        return ""
    elif word == "dont":
        return "do"
    elif word == "don't":
        return "do"
    elif word == "cant":
        return "can"
    elif word == "can't":
        return "can"
    elif word == "wont":
        return "will"
    elif word == "won't":
        return "will"
    elif word == "shouldnt":
        return "should"
    elif word == "can":
        return "can't"
    elif word == "should":
        return "shouldn't"
    elif word == "will":
        return "won't"
    elif word == "but":
        return "and"
    elif word == "and":  # Check the tags!
        return "or"
    elif word == "or":
        return "and"
    else:
        return word

# Words is the tagged output of parse_tag
# Target is either 1 (positive) or -1 (negative)
def Translate(words, target):
    new_clauses = []
    #print words
    n_clause = len(words)
    i_clause = 0
    for clause in words:
        i_clause+= 1
        print "optimizing clause %i out of %i"%(i_clause, n_clause)
        new_clause = [word[0] for word in clause]
        best_sentiment = sentiment(" ".join(new_clause))
        print best_sentiment
        for i in xrange(len(clause)):
            word = clause[i][0]
            tag = clause[i][1]
            s_tag = wn_tag(tag)

            # Invert stopwords as necessisary
            new_clause[i] = invert_stopwords(strip(word))

            if word not in stopwords.words() and word == new_clause[i]:
                # Preserve trailing punctuation 
                punc = []
                if word[-1] in string.punctuation or word[-1] not in string.printable:
                    punc = [c for c in word if c in string.punctuation or c not in string.printable]

                # Replace word with antonym
                anti_found = False
                for w in wn.synsets(strip(word), pos=s_tag):
                    for j in w.lemmas():
                        if j.antonyms():

                            #Conjugate words???
                            new_clause[i] = " ".join(j.antonyms()[0].name().split("_"))
                            if len(punc) > 0:
                                new_clause[i] += "".join(punc)
                            anti_found = True
                            break
                    if anti_found:
                        break

                # If the word has no antonym, add an inverter
                if not anti_found and (s_tag == wn.ADJ or s_tag == wn.ADV):
                    new_clause[i] = "not " + word

                # Invert verbs (make tense sensitive?)
                elif not anti_found and s_tag == wn.VERB:
                    new_clause[i] = "didn't " + word

            # Check if the sentiment has gone in the right direction
            new_sentiment = sentiment(" ".join(new_clause))
            change_score = (best_sentiment - new_sentiment) * target
            print new_sentiment
            if change_score < 0: # If not, revert the change.
                new_clause[i] = word
            else: 
                best_sentiment = new_sentiment
                #print best_sentiment

            


        new_clauses.append(new_clause)
    return new_clauses




@app.route('/handle_data', methods=['POST'])
def handle_data():
    text = request.form['input_text']
    target = request.form['sentiment']
    if target == 'pos' or target == 'p' or target == '1' or target == '+':
        target = -1
    else:
        target = 1

    tags = parse_tag(text)

    valid = is_valid(text)

    #train_model()

    new_words = Translate(tags, target)

    output = " ".join(" ".join(clause) for clause in new_words)
    print output
    return output



if __name__ == '__main__':
    app.run(debug=True)


# Read and format data

# Analyze sentiment

# While sentiment doesn't match desired sentiment:


    # Find the word that most strongly contributes to the sentiment

    # Find the closest antonym to that word

    # Check if sentence is still valid

    # If not, or no antonym found, and word is an adjective, try adding "not"

    # Check if sentence is still valid

    # If not, or no antonym found, and word is a verb, try adding "did not" or "do not"