from nltk.corpus import wordnet
from xml.dom import minidom

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


def getTree(wnid, isa_tuple, words_dict):
    layer = 0
    word_tree = [] 
 
    
    while(True):

        layer += 1
        if wnid not in isa_tuple:
            break 
  
        p_wnid = isa_tuple[wnid]
        p_names = words_dict[p_wnid]
        word_tree.append((layer, p_wnid, p_names)) 
        wnid = p_wnid

    return word_tree

if __name__ == "__main__":
    
    synset_file = "/home/t-yuche/caffe/data/ilsvrc12/synset_words.txt"
    ##
    words_file = 'words.txt'
    isa_file = 'wordnet.is_a.txt'
    
    ##
    isa_tuple = {}
    with open(isa_file) as f:
        for l in f.readlines():
            segs = l.strip().split(' ')
            isa_tuple[segs[1]] = segs[0]
    ##   
    words_dict = {} 
    with open(words_file) as f:
        for l in f.readlines():
            segs = l.strip().split('\t')
            wnid = segs[0]
            names = segs[1]
            words_dict[wnid] = names
   
    word_tree = {} 
    with open(synset_file) as f:
       for l in f.readlines():
            wnid = l.strip().split(' ')[0]
            names = [x.strip() for x in ' '.join(l.strip().split(' ')[1:]).split(',')]
            word_tree[wnid] = [getTree(wnid, isa_tuple, words_dict)]
     
    #preprocessSynsetWords(synset_file)
    # dump in json
    json.dump(word_tree, open('synset_word_tree', 'w'))
