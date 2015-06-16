from nltk.corpus import wordnet

import pandas as pd
import sys


def getHypernyms(words):
     
    words = [ '_'.join(word.strip().split(' ')) for word in words]
    print words

    synset_list = []    
    for word in words:
        synset_list += [wordnet.synsets(word)]
               
    syn = list(set.intersection(*map(set, synset_list)))[0]
    
    print syn.hypernyms()


def preprocessSynsetWords(file_name):
    
    with open(file_name) as f:
             
        labels_df = pd.DataFrame([
        {
            'synset_id':l.strip().split(' ')[0],
            'name': ' '.join(l.strip().split(' ')[1:]).split(',')
        }
        for l in f.readlines()
    ])
    
    labels = labels_df.sort('synset_id')['name'].values


    
    label_match = {}
    for label in labels:
         
        getHypernyms(label)
        
    #print label_match


if __name__ == "__main__":
    
    synset_file = "/home/t-yuche/caffe/data/ilsvrc12/synset_words.txt"
    preprocessSynsetWords(synset_file)
    # dump in json
