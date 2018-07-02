# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 18:04:55 2018

@author: HULK
"""

import nltk
from nltk.corpus import wordnet


def findSynonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms               


def findSimilarity(word1,word2):
     verbs = {x.name().split('.', 1)[0] for x in wordnet.all_synsets('v')}
     if(word1 in verbs):
         word1+=(".v.01")
     else:
         word1+=(".n.01")
     if(word2 in verbs):
         word2+=(".v.01")
     else:
         word2+=(".n.01")
         
     w1 = wordnet.synset(word1)
     w2 = wordnet.synset(word2)     
     return w1.wup_similarity(w2)
 
    