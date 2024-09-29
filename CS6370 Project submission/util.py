# Add your import statements here
import numpy as np
import nltk
from nltk.tokenize import PunktSentenceTokenizer
import re
import argparse
import json
from sys import version_info
import time
nltk.download('punkt')
# Add any utility functions here

parser = argparse.ArgumentParser(description='main.py')
parser.add_argument('-dataset', default="cranfield/", help="Path to the dataset folder")
args, unknown_args = parser.parse_known_args()

if unknown_args:
    print("Warning: Unrecognized command-line arguments:", unknown_args)

docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:][:]
docs = [item["body"] for item in docs_json]
#vocab is the vocabulary of the dataset
vocab=[]
##all of the above code is just to load the cranefield dataset without paths

def domSpec_stopword(T):
    n=[]
    #computes the number of documents a word has occured 
    #the indices of n and vocab list are consistent
    docs_words=[[]]
    for i in range(len(docs)):
        temp_list=re.split(r'[ ]',docs[i])
        #each document is directly tokenized without any other preprocessing
        for j in range(len(temp_list)):
                if temp_list[j] not in vocab:
                    #if seeing a word for the first time
                    vocab.append(temp_list[j])
                    n.append(1)
                    #reason for consistency
                else:
                    ind=vocab.index(temp_list[j])
                    #because of consistency
                    #update the value of n
                    if n[ind]<i+1:
                        #i+1 because we don't want the number of times a word occurs in a document but the 
                        #number of docs it appears in
                        n[ind]+=1
        docs_words.append(temp_list)
    #creating a bag of words(list) for each document and storing them in a list(doc_words)
    docs_words=docs_words[1:]
    #removing the first term used for initialisation
    stopwords=[]
    #list of stop words
    for i in range(len(n)):
        if n[i] > T:
            stopwords.append(vocab[i])
    
    # print(stopwords)

domSpec_stopword(700)
#Prints the common words between stopwords we get from nltk and the vocab of the corpus to compare this to the domspec list
stopwords_nltk=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
intersection_result = [value for value in vocab if value in stopwords_nltk]
# print("Stop words we can obtain using NLTK that are present in our dataset are: ")
# print(intersection_result)









