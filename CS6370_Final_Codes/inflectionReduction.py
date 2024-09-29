from util import *

# Add your import statements here
from nltk.stem import *
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *




class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = None
		lemmatizer = WordNetLemmatizer()
		my_list=[]
		stemmer = PorterStemmer()
		#load the porterstemmer 
		for i in range(len(text)):
			#text[i] is a sentence
			#t is a token in the original sentence
			temp_list=[lemmatizer.lemmatize(t) for t in text[i]]
			#returns a list of stemmed tokens
			my_list.append(temp_list)
		reducedText=my_list


		#Fill in code here
		
		return reducedText


