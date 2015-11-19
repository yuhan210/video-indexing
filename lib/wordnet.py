import json


def includerelation():
    includerelation = {'musical instrument': 'guitar'}
    return includerelation

def convert_to_equal_word():

    convert_dict = {'domestic cat': 'cat', 'people': 'person', 'man': 'person', 'men': 'person', 'woman': 'person', 'women': 'person', 'girl': 'person', 'boy': 'person', 'lady': 'person', 'kid': 'person', 'baby': 'person', 'skiing': 'ski'}
    
    return convert_dict 

def word_wnid_dict():
    
    word_to_wnid_dict = {}
    wnid_to_word_dict = {}
    with open('/home/t-yuche/lib/wordnet_data/synset_words.txt') as f:
        for l in f:
            wnid = l.strip()[:9]
            s = l.strip()[10:]
            word_to_wnid_dict[s] = wnid
            wnid_to_word_dict[wnid] = s

    return word_to_wnid_dict, wnid_to_word_dict

def word_prefix_to_wnid_dict():

    word_pref_to_wnid = {}

    with open('/home/t-yuche/lib/wordnet_data/synset_words.txt') as f:
        for l in f:
            wnid = l.strip()[:9]
            s = l.strip()[10:]
            w_pref = s.split(',')[0]
            word_pref_to_wnid[w_pref] = wnid
    
    return word_pref_to_wnid 

def wnid_to_label_dict():
  
    wnidtolabel = {} 
    with open('/home/t-yuche/lib/wordnet_data/train_caffe.txt') as f:
        for l in f:
            segs = l.strip().split()
            caffe_label = int(segs[1])
            wnid = segs[0].split('/')[0]
            wnidtolabel[wnid] = caffe_label

    return wnidtolabel

def wnid_to_stopword_dict():

    wnid_to_stopword = {}
    
    dummy, wnidtow = word_wnid_dict()
    wtos = word_to_stopword_dict()    

    for wnid in wnidtow:
        w = wnidtow[wnid]
        wnid_to_stopword[wnid] = wtos[w]

    return wnid_to_stopword 

def word_pref_to_stopword_pref_dict():
    
    wordp_to_stopwordp = {}
    
    with open('/home/t-yuche/lib/wordnet_data/new_synset_word.txt') as f:
        data = json.load(f)
        for w in data:
            stop_word = data[w]
            if type(stop_word) is list:
                stop_word = ', '.join(stop_word)
            w = w.replace('_', ',').split(',')[0]
            wordp_to_stopwordp[w] = stop_word.split(',')[0]

    return wordp_to_stopwordp
   

def word_to_stopword_dict():
   
    word_to_stopword = {}
    
    with open('/home/t-yuche/lib/wordnet_data/new_synset_word.txt') as f:
        data = json.load(f)
        for w in data:
            stop_word = data[w]
            if type(stop_word) is list:
                stop_word = ', '.join(stop_word)
            w = w.replace('_', ',')
            
            word_to_stopword[w] = stop_word

    return word_to_stopword
   
def wnid_traces_dict():

    dummy, wnidtow = word_wnid_dict()
    wtrace = word_traces_dict()

    trace = {} 
    for wnid in wnidtow:
        trace[wnid] = wtrace[wnidtow[wnid]]    
     
    return trace

'''
A key to list mapping
The list describes how the word climb to the stopword
'''
def word_traces_dict():

    wtostop = word_to_stopword_dict()
    wtownid, dummy = word_wnid_dict()

    with open('/home/t-yuche/lib/wordnet_data/synset_word_tree.txt') as f:
        word_tree = json.load(f)

    
    trace = {}
    for word in wtostop:
        wnid = wtownid[word]
        stop_word = wtostop[word]
        trace[word] = [[word]]
        if stop_word != word:
            
            for iw in word_tree[wnid][0]:
                iw = iw[-1]
                trace[word] += [[iw]]
                if iw == stop_word:
                    break

    return trace

