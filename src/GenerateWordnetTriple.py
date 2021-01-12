from nltk.corpus import wordnet as wn
from tqdm import tqdm

writer=open(r'/home/shcd/knowledge enhanced language model/data/wordnet_sibling.txt','w',encoding='utf-8')

def get_hyponyms(synset):
    synset_hyponym_list = list(set([synset_hyponym for synset_hyponym in synset.hyponyms()]))
    ret={}
    for synset_hyponym in synset_hyponym_list:
        for name in synset_hyponym.lemma_names():
            if name=='woman':
                print('cwy debug')
            ret[name]=1
    return ret
    hyponyms = set()
    for hyponym in synset.hyponyms():
        hyponyms.add(hyponym)# |= set(get_hyponyms(hyponym))
    return hyponyms #| set(synset.hyponyms())


lemmas_in_words  = list(set(i for i in wn.words()))


#Generate all possible hyponym, including synset
'''for word in tqdm(lemmas_in_words,total = len(lemmas_in_words)):
    if word.startswith('.'):
        continue
    synset_set={}
    hyponym_set={}
    for synset in wn.synsets(word):
        for lemma in synset.lemma_names():
            synset_set[lemma.lower()]=1
        hyponym_this=get_hyponyms(synset)
        for hyp in hyponym_this:
            hyponym_set[hyp.lower()]=1
    a=1
    for syn in synset_set:
        if word!=syn:
            writer.write('{}\tsynset\t{}\n'.format(word.lower(),syn))
    for hyp in hyponym_set:
        if hyp!=word:
            writer.write('{}\thyponym\t{}\n'.format(word.lower(),hyp))
            writer.write('{}\thypernym\t{}\n'.format(hyp, word.lower()))
writer.close()'''

for word in tqdm(lemmas_in_words,total = len(lemmas_in_words)):
    if word.startswith('.'):
        continue
    for pos_str,pos in [['N',wn.NOUN], ['J',wn.ADJ], ['V',wn.VERB], ['R',wn.ADV]]:
        synset_set = {}
        hyponym_set = {}
        for synset in wn.synsets(word,pos):
            for lemma in synset.lemma_names():
                synset_set[lemma.lower()]=1
            hyponym_this=get_hyponyms(synset)
            for hyp in hyponym_this:
                hyponym_set[hyp.lower()]=1
        a=1
        for syn in synset_set:
            if word!=syn:
                writer.write('{}\tsynset\t{}\t{}\n'.format(word.lower(),syn,pos_str))
        for hyp in hyponym_set:
            if hyp!=word:
                writer.write('{}\thyponym\t{}\t{}\n'.format(word.lower(),hyp,pos_str))
                writer.write('{}\thypernym\t{}\t{}\n'.format(hyp, word.lower(),pos_str))
writer.close()


