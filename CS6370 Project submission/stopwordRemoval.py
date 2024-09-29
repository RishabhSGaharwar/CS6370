from util import *

# Add your import statements here
import nltk
from nltk.corpus import stopwords




class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = None
		# nltk.download('stopwords')
		from nltk.corpus import stopwords
		stop_words = list(stopwords.words('english'))
		#import the stop words needed for removal
		my_list=[]
		for i in range(len(text)):
			#remove if a word is present in the above list
			temp_list=[w for w in text[i] if w.lower() not in stop_words]
			my_list.append(temp_list)
		
		stopwordRemovedText=my_list
		return stopwordRemovedText

		
		
		




	