from util import *
from math import log10, sqrt
import numpy as np


# Add your import statements here




class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.doc_IDs = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable
		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		index = None
		# Initialize index as a nested dictionary with types as keys and empty dictionaries as values
		index={tokens: {} for d in docs for sentence in d for tokens in sentence}
		# List to flatten the documents into a single list of tokens
		flattened_docs=[]
        
		# Iterate through each document, sentence, and token to build the flattened_docs list
		for doc in docs:
			flattened_docs.append([token for sentence in doc for token in sentence])

        # Iterate through each document, sentence, and term to update the index dictionary
		for doc_ind in range(len(docs)):
			for sentence in docs[doc_ind]:
				for term in sentence:
					# Check if the document ID is not already in the index for the current term
					if docIDs[doc_ind] not in index[term]:
						# Count the frequency of the term in the flattened document and update the index
						index[term][docIDs[doc_ind]]=flattened_docs[doc_ind].count(term)

		# Store the index and document IDs in the class attributes
		self.index = index
		self.doc_IDs = docIDs


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []
		# Retrieve index and document IDs from class attributes
		index=self.index
		docIDs=self.doc_IDs

		# Create a list of unique terms and get the counts of documents and terms
		term_list=list(index.keys())
		doc_count=len(docIDs)
		term_count=len(term_list)

		# Calculate inverse document frequency (IDF) for each term and store them 
		#in a dictionary with unique terms as keys
		idf={}
		for term in term_list:
			idf[term]=log10(len(docIDs)/len(index[term]))
		
		# Initialize a matrix to store the tf-idf values
		doc_mat=np.zeros((term_count,doc_count))
		
		# Populate the term-document matrix with weighted term frequencies
		for i in range(term_count):
			for j in range(doc_count):
				# Check if the current document ID exists in the index for the current term
				#if not present tf-idf=0 for that document ID , current term
				if j+1 in index[term_list[i]]:
					# Multiply the term frequency by the inverse document frequency (IDF)
					doc_mat[i][j]= index[term_list[i]][j+1] * idf[term_list[i]]
		# Transpose the matrix for further processing
		doc_mat_transpose=doc_mat.T

		for query in queries:
			# Flatten the query into a list of tokens
			flattened_query=[token for sentence in query for token in sentence]

			# Initialize a vector to represent the query
			query_vec=np.zeros(term_count)

			# Calculate the tf*idf values for each term for the query vector
			for i in range(term_count):
				query_vec[i]=flattened_query.count(term_list[i])*idf[term_list[i]]
			
			# Initialize a dictionary to store document similarities
			# With keys as documentIDs and values are cosine sim values
			sim_dict={}

			# Calculate cosine sim of the query with each document and store it in the above dict
			vector2 = np.array(query_vec)
			magnitude2 = np.linalg.norm(vector2)
			for doc_id in range(doc_count):
				vector1 = np.array(doc_mat_transpose[doc_id])
				dot_product = np.dot(vector1, vector2)
				magnitude1 = np.linalg.norm(vector1)
				if dot_product>0:
					sim_dict[doc_id+1]=dot_product / (magnitude1 * magnitude2)
				else:
					sim_dict[doc_id+1]=0
			
			# Sort document similarities in descending order
			sorted_tuples = sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)
			# Obatin the keys (i.e documentIDs with similarities in descending order)
			sorted_keys = [key for key, value in sorted_tuples]

			# Append ordered document IDs to the list, done for each query
			doc_IDs_ordered.append(sorted_keys)

		# Return the list of ordered document IDs for each query
		return doc_IDs_ordered