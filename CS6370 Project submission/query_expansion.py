from util import *
from math import log10, sqrt
import numpy as np
from scipy.linalg import svd

from nltk.corpus import wordnet

class query_expansion():

    def expander(self,queries):
        mod_queries=queries
        for i in range(len(queries)):
            for j in range(len(queries[i])):
                sent_expansion=[]
                for term in queries[i][j]:
                    term=term.lower()
                    synsets=wordnet.synsets(term)
                    if len(synsets)>0:
                        for ii in range(1):
                            sent_expansion.append(synsets[ii].name().split('.')[0])
                    else:
                        if '-' in term:
                            words=term.split("-")
                            word1=words[0]
                            word2=words[1]
                            synsets1=wordnet.synsets(word1)
                            synsets2=wordnet.synsets(word2)
                            if len(synsets1)>0:
                                sent_expansion.append(synsets1[0].name().split('.')[0])
                            if len(synsets2)>0:
                                sent_expansion.append(synsets2[0].name().split('.')[0])
                mod_queries[i][j]=queries[i][j]+sent_expansion
                
        return mod_queries
    

                
        
