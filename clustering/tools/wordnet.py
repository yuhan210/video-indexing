from nltk.corpus import wordnet

import pandas as pd
import sys
import json
import os

def getHypernyms(words):
     
    words = [ '_'.join(word.strip().split(' ')) for word in words]

    synset_list = []    
    for word in words:
        synset_list += [wordnet.synsets(word)]
               
    syns = list(set.intersection(*map(set, synset_list)))

    best_words = words
    
    ## pick one
   
     
    print best_words
    while(True):
 
        ''' printing '''
        for idx, syn in enumerate(syns):
            print idx, syn, syn.definition()
        print ''
   
        ''' picking ''' 
        syn_ind = raw_input('pick one syn (0)')
        if len(syn_ind) == 0:
            syn_ind = 0

        syn = syns[int(syn_ind)]
        print 'pick syn:', syn, syn.lemma_names()
        print ''
        ## whether go up
        ''' picking '''
        go_up_str = raw_input('go up? (y/(n))')       
        if len(go_up_str) == 0 or go_up_str == 'y':
            syns = syn.hypernyms()
        else:
            best_words = syn.lemma_names()
            print  words, ' to ' , best_words , ' '
            
            break
    
    
    return best_words
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
     
    # load things labels from json file
    
    save_file = 'new_synset_word.txt'
    new_labels = {}
    if os.path.exists(save_file):
        new_labels = json.load(open(save_file))
 
    for idx, label in enumerate(labels):
        key = '_'.join(label)
        if key not in new_labels: 
            new_label = getHypernyms(label)
            print label, new_label 
            new_labels[key] = new_label
        # dump result to json file
        json.dump(new_labels, open(save_file, 'w'))
  
    #print label_match


if __name__ == "__main__":
    
    synset_file = "/home/t-yuche/caffe/data/ilsvrc12/synset_words.txt"
    
    preprocessSynsetWords(synset_file)
    # dump in json
