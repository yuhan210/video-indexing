from nltk.corpus import wordnet
import sys

def getHypernyms(word):
    syn = wordnet.synsets(word)[0]
    
    # get the list of hypernyms
    return syn.hypernyms()[0].hypernyms()[0].hypernyms()

if __name__ == "__main__":
    
   print getHypernyms(sys.argv[1]) 
