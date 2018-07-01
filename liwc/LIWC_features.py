import nltk
import sys
from liwc.LIWC_features_dictionary import get_dictionary
from liwc.tokenizer import get_word_only_token

import timeit


def get_liwc_features(document:str) -> (dict, int):
    
    #t1= timeit.default_timer()
    
    dictionary = get_dictionary()
    
    #t2= timeit.default_timer()
    
    word_only_list = get_word_only_token(document)
    feature_dict = {}
    
    freq_dist = nltk.FreqDist(word_only_list)

    for category, keyworks in dictionary.items():
        remaining_keywords = []
        total = 0

        for keyword in keyworks:
            if keyword.endswith("*"):
                remaining_keywords.append(keyword[:len(keyword)-1])
            elif keyword in freq_dist:
                total += freq_dist[keyword]

        for word, freq in freq_dist.items():
            for keyword in remaining_keywords:
                if word.startswith(keyword):
                    total += freq

        feature_dict[category] = total
        
   # t3= timeit.default_timer()
    
    #print("delta times t21, t32: ", t2 - t1, ", ", t3 - t2)
        
    return (feature_dict, len(word_only_list))




    

    
            
            