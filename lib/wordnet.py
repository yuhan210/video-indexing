import json



def word_wnid_dict():
    
    word_to_wnid_dict = {}
    wnid_to_word_dict = {}
    with open('./wordnet_data/synset_words.txt') as f:
        for l in f:
            wnid = l.strip()[:9]
            s = l.strip()[10:]
            word_to_wnid_dict[s] = wnid
            wnid_to_word_dict[wnid] = s

    return word_to_wnid_dict, wnid_to_word_dict


def wnid_to_stopword_dict():

    wnid_to_stopword = {}
    
    dummy, wnidtow = word_wnid_dict()
    wtos = word_to_stopword_dict()    

    for wnid in wnidtow:
        w = wnidtow[wnid]
        wnid_to_stopword[wnid] = wtos[w]

    return wnid_to_stopword 

def word_to_stopword_dict():
   
    word_to_stopword = {}
    
    with open('./wordnet_data/new_synset_word.txt') as f:
        data = json.load(f)
        for w in data:
            stop_word = data[w]
            if type(stop_word) is list:
                stop_word = ', '.join(stop_word)
            w = w.replace('_', ',')
            
            word_to_stopword[w] = stop_word

    return word_to_stopword
   
'''
A key to list mapping
The list describes how the word climb to the stopword
'''
def word_traces_dict():

    wtostop = word_to_stopword_dict()
    wtownid, dummy = word_wnid_dict()

    with open('./wordnet_data/synset_word_tree.txt') as f:
        word_tree = json.load(f)

    
    trace = {}
    for word in wtostop:
        wnid = wtownid[word]
        stop_word = wtostop[word]
        trace[word] = [[word]]
        if stop_word != word:
            
            for iw in word_tree[wnid][0]:
                iw = iw[-1]
                print iw
                trace[word] += [[iw]]
                if iw == stop_word:
                    break

    return trace

