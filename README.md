# sentimenttranslator

You will need to download the stanford POS tagger from:
https://nlp.stanford.edu/software/tagger.html

into the "stanford" folder. Also depends on Flask.


The sentiment translator will attempt to invert the sentiment of the input text. To run, run "python sentiment_translator.py", then open the ui at localhost:5000
The large box accepts input, the small box sets the target sentiment: "p", "pos", "+", or "1" for positive, "n", "neg", "-", or "-1" for negative. 