import nltk
from nltk.stem.wordnet import WordNetLemmatizer

text = nltk.word_tokenize("a man in a blue shirt is sitting on a wooden bench with his arms outstretched")


#segs = nltk.word_tokenize(sentence)
tags = nltk.pos_tag(text)
print tags
verbs = []
for word, tag in tags:
    if tag == 'VB' or tag == 'VBZ' or tag == 'VBN' or tag == 'VBD' or tag == 'VBG':
        verbs += [str(WordNetLemmatizer().lemmatize(word, 'v'))]

print verbs
