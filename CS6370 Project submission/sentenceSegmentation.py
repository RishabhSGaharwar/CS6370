from util import *

# Add your import statements here
import nltk
from nltk.tokenize import PunktSentenceTokenizer
import nltk
nltk.data.path.append("/path/to/nltk_data")
import re




class SentenceSegmentation():

	def naive(self, text):

		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None
		#text is a sentence which is split whenever . / ? / ! occurs
		new_list=re.split(r'[.?!]',text)
		new_list=[sentence.strip() for sentence in new_list if sentence.strip()]
		#removing extra de-limiters and putting the senetences in a list.
		segmentedText=new_list
		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = None
		#download the punkt tokenizer
		
		custom_sent_tokenizer = PunktSentenceTokenizer(text)
		tokenized_sentences = custom_sent_tokenizer.tokenize(text)
		#returns a list of sentences
		segmentedText=tokenized_sentences
		
		return segmentedText