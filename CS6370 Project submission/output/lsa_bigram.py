from util import *
from math import log10, sqrt
import numpy as np
from scipy.linalg import svd


# Add your import statements here




class lsa():

	def __init__(self):
		self.index = None
		self.doc_IDs = None
		self.lsi=None

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
		index={tokens: {} for d in docs for sentence in d for tokens in sentence}
		doc_count = len(docs)
		flattened_docs=[]

		for doc in docs:
			flattened_docs.append([token for sentence in doc for token in sentence])


		for doc_ind in range(len(docs)):
			for sentence in docs[doc_ind]:
				for term in sentence:
					if docIDs[doc_ind] not in index[term]:
						index[term][docIDs[doc_ind]]=flattened_docs[doc_ind].count(term)
		
		# self.index = index
		# self.doc_IDs = docIDs
		bigram_dic={}
		for doc_ind in range(len(docs)):
			id=docIDs[doc_ind]
			for sentence_ind in range(len(docs[doc_ind])):
				sentence=docs[doc_ind][sentence_ind]
				sent_len=len(sentence)
				j=0
				while j < sent_len - 1:
					tokens=[]
					for i in range(j,j+2):
						tokens.append(sentence[i])
					tokens=tuple(tokens)
					if tokens not in index:
						bigram_dic[tokens]={id:1}
					else:
						idVals = bigram_dic.get(tokens)
						idVals[id] = idVals.get(id, 0) + 1
					j+=1
		sum_dic={}
		for bigram in bigram_dic:
			sum=0
			for doc_id in bigram_dic[bigram]:
				sum+=bigram_dic[bigram][doc_id]
			sum_dic[bigram]=sum
		sorted_sumdict = sorted(sum_dic, key=sum_dic.get, reverse=True)[:(len(sum_dic)*(15))//100]
		for key in sorted_sumdict:
			index[key] = bigram_dic[key]	
		
		term_list=list(index.keys())
		term_count=len(term_list)
		
		td_matrix=np.zeros((term_count,doc_count))
		for i in range(term_count):
			for j in range(doc_count):
				if (j+1) in index[term_list[i]]:
					td_matrix[i][j] = index[term_list[i]][j+1]
				
		self.index = index
		self.doc_IDs = docIDs
		k=500
		U,S,V=svd(td_matrix)
		U1 = U[:,:k]
		S1 = S[:k]
		V1 = V[:k]
		D1 = np.diag(S1)
		td_matrix_k = U1 @ D1 @ V1
		lsi={tokens: {} for d in docs for sentence in d for tokens in sentence}
        
		for i in range(term_count):
			for j in range(doc_count):
				lsi[term_list[i]][j+1]=td_matrix_k[i][j]
        
		self.lsi=lsi
				
		
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
		lsi=self.lsi

		term_list=list(index.keys())
		doc_count=len(docIDs)
		term_count=len(term_list)
		idf={}
		
		for term in term_list:
			idf[term]=log10(len(docIDs)/len(index[term]))
		
		doc_mat=np.zeros((term_count,doc_count))
		
		for i in range(term_count):
			for j in range(doc_count):
				if j+1 in lsi[term_list[i]]:
					doc_mat[i][j]= lsi[term_list[i]][j+1] * idf[term_list[i]]

		doc_mat_transpose=doc_mat.T

		for query in queries:
			flattened_query=[token for sentence in query for token in sentence]
			query_ngram={}
			for sentence in query:
				sent_len=len(sentence)
				j=0
				while j < sent_len - 1:
					tokens=[]
					for i in range(j,j+2):
						tokens.append(sentence[i])
					tokens=tuple(tokens)
					if tokens not in query_ngram:
						query_ngram[tokens]=1
					else:
						query_ngram[tokens]+=1
					j+=1
			query_vec=np.zeros(term_count)
			for i in range(term_count):
				if type(term_list[i]) == tuple:
					if term_list[i] in query_ngram:
						query_vec[i]=query_ngram[term_list[i]]*idf[term_list[i]]
					else:
						query_vec[term_list[i]]=flattened_query.count(term_list[i])*idf[term_list[i]]
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