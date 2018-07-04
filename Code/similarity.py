# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 18:04:55 2018

@author: HULK
"""

import nltk
from nltk.corpus import wordnet

#find synonyms
def findSynonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms               

#find similarity between 2 words check and append if it is noun or verb
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
 
def Average(list1):
    return sum(list1) / len(list1)
     
#returns accuracy list1 is predicted list 2 is original
 
def accuracy(list1,list2):
    finalList=[]
    for word1 in list1:
        listcheck=[]
        for word2 in list2:
            k=findSimilarity(word1,word2)
            listcheck.append(k)
        finalList.append(min(listcheck))
    return 1-Average(finalList)




def finalAccuracy(list1,list2):
    list1=[element.lower() for element in list1]
    list2=[element.lower() for element in list2]
    k=list(set(list1).intersection(list2))
    ac1=len(k)/len(list1)
    temp1=list(set(list1)-set(k))
    temp2=list(set(list2)-set(k))
    ac2=accuracy(temp1,temp2)
    return(ac1+ac2)/2
    
list1=["Book","School","bus","aeroplane","computer"]
list2=["computer","Airport","Pen","School","car"]    
print(finalAccuracy(list1,list2))    
