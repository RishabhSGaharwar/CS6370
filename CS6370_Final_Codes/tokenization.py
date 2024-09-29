from util import *

# Add your import statements here
from nltk.tokenize.treebank import TreebankWordTokenizer




class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None
		my_list=[]
		for i in range(len(text)):
			#text[i] is a sentence
			temp_list=re.split(r'[,// ]',text[i])
			my_list.append(temp_list)
		tokenizedText=my_list
		#we traverse the list and tokenize each sentence
		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None
		my_list=[]
		t = TreebankWordTokenizer()
		#load the ptb tokenizer
		for i in range(len(text)):
			temp_list=t.tokenize(text[i])
			#returns a list of tokens for the sentence text[i]
			my_list.append(temp_list)
		tokenizedText=my_list

		#Fill in code here

		return tokenizedText