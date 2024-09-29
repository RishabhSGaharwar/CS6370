from util import *
from math import log10, sqrt
import numpy as np


# Add your import statements here




class ngram():

	def __init__(self,n):
		self.index = None
		self.doc_IDs = None
		self.n=n

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
		n = self.n
		index={}
		total_docs = len(docs)

		for doc_ind in range(len(docs)):
			id=docIDs[doc_ind]
			for sentence_ind in range(len(docs[doc_ind])):
				sentence=docs[doc_ind][sentence_ind]
				sent_len=len(sentence)
				j=0
				while j < sent_len - n + 1:
					tokens=[]
					for i in range(j,j+n):
						tokens.append(sentence[i])
					tokens=tuple(tokens)
					if tokens not in index:
						index[tokens]={id:1}
					else:
						idVals = index.get(tokens)
						idVals[id] = idVals.get(id, 0) + 1
					j+=1
		#Fill in code here

		self.index = index
		self.doc_IDs = docIDs
		self.n=n


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

		index=self.index
		docIDs=self.doc_IDs
		n=self.n

		term_list=list(index.keys())
		doc_count=len(docIDs)
		term_count=len(term_list)
		idf={}
		
		for term in term_list:
			idf[term]=log10(len(docIDs)/len(index[term]))
		
		doc_mat=np.zeros((term_count,doc_count))
		
		for i in range(term_count):
			for j in range(doc_count):
				if j+1 in index[term_list[i]]:
					doc_mat[i][j]= index[term_list[i]][j+1] * idf[term_list[i]]

		doc_mat_transpose=doc_mat.T

		for query in queries:
			flattened_query=[token for sentence in query for token in sentence]
			query_vec=np.zeros(term_count)
			query_ngram={}
			for sentence in query:
				sent_len=len(sentence)
				j=0
				while j < sent_len - n + 1:
					tokens=[]
					for i in range(j,j+n):
						tokens.append(sentence[i])
					tokens=tuple(tokens)
					if tokens not in query_ngram:
						query_ngram[tokens]=1
					else:
						query_ngram[tokens]+=1
					j+=1
			for i in range(term_count):
				if term_list[i] in query_ngram:
					query_vec[i]=query_ngram[term_list[i]]*idf[term_list[i]]
			sim_dict={}
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
			sorted_tuples = sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)
			sorted_keys = [key for key, value in sorted_tuples]

			doc_IDs_ordered.append(sorted_keys)
		return doc_IDs_ordered