from util import *

# Add your import statements here

from numpy import log2 as log2


class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1
		if(len(query_doc_IDs_ordered)==0):
			return precision
		precision = 0
		relevantDocs = []
		for entry in true_doc_IDs:
			if int(query_id)==int(entry['query_num']):
				relevantDocs.append(int(entry['id']))
		for i in range(min(k, len(query_doc_IDs_ordered))):
			docID = query_doc_IDs_ordered[i]
			if int(docID) in relevantDocs:
				precision += 1
			# for entry in true_doc_IDs:
			# 	if int(entry['id'])==int(docID) and int(query_id)==int(entry['query_num']):
			# 		precision += 1
		precision /= k
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1
		if(len(doc_IDs_ordered)!=len(query_ids) or len(query_ids)==0):
			return meanPrecision
		meanPrecision = 0
		for i in range(len(query_ids)):
			meanPrecision += self.queryPrecision(doc_IDs_ordered[i], query_ids[i], qrels, k)
		meanPrecision /= len(query_ids)
		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1
		if(len(query_doc_IDs_ordered)==0):
			return recall
		total_relevant = 0
		relevantDocs = []
		for entry in true_doc_IDs:
			if int(query_id)==int(entry['query_num']):
				relevantDocs.append(int(entry['id']))
		total_relevant = len(relevantDocs)
		if total_relevant==0:
			return recall
		recall = 0
		for i in range(min(k, len(query_doc_IDs_ordered))):
			docID = query_doc_IDs_ordered[i]
			if int(docID) in relevantDocs:
				recall += 1
		recall /= total_relevant
		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1
		if(len(doc_IDs_ordered)!=len(query_ids) or len(query_ids)==0):
			return meanRecall
		meanRecall = 0
		for i in range(len(query_ids)):
			meanRecall += self.queryRecall(doc_IDs_ordered[i], query_ids[i], qrels, k)
		meanRecall /= len(query_ids)
		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k, beta=1):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		if recall==0 or precision==0:
			return 0
		fscore = (beta*beta+1)*precision*recall
		fscore /= ((beta*beta)*precision + recall)
		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k, beta=1):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1
		if(len(doc_IDs_ordered)!=len(query_ids) or len(query_ids)==0):
			return meanFscore
		meanFscore = 0
		for i in range(len(query_ids)):
			meanFscore += self.queryFscore(doc_IDs_ordered[i], query_ids[i], qrels, k, beta)
		meanFscore /= len(query_ids)
		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1
		relevance_scores = {}
		for entry in true_doc_IDs:
			if int(entry['query_num']) == int(query_id):
				relevance_scores[int(entry['id'])] = 5 - int(entry['position'])
		DCG = 0
		for i in range(min(k, len(query_doc_IDs_ordered))):
			docID = int(query_doc_IDs_ordered[i])
			rel = relevance_scores.get(docID) if relevance_scores.get(docID) is not None else 0
			DCG += (rel)/(log2(i+2))
		IDCG = 0
		sorted_scores = sorted(relevance_scores.values(), reverse=True)
		for i in range(min(k, len(sorted_scores))):
			rel = sorted_scores[i]
			IDCG += rel/log2(i+2)
		if(IDCG==0):
			print("IDCG is zero. Retuning -1")
			return -1
		nDCG = DCG/IDCG
		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1
		if(len(doc_IDs_ordered)!=len(query_ids) or len(query_ids)==0):
			return meanNDCG
		meanNDCG = 0
		for i in range(len(query_ids)):
			meanNDCG += self.queryNDCG(doc_IDs_ordered[i], query_ids[i], qrels, k)
		meanNDCG /= len(query_ids)
		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1
		if(len(query_doc_IDs_ordered)==0):
			return avgPrecision
		relevantDocCount = 0
		avgPrecision = 0
		relevantDocs = []
		for entry in true_doc_IDs:
			if int(query_id)==int(entry['query_num']):
				relevantDocs.append(int(entry['id']))
		for i in range(min(k, len(query_doc_IDs_ordered))):
			docID = int(query_doc_IDs_ordered[i])
			if docID in relevantDocs:
				relevantDocCount += 1
				avgPrecision += relevantDocCount/(i+1)
		if relevantDocCount==0:
			# print(f"No relevant documents in MAP for query {query_id} and k = {k}")
			return 0
		avgPrecision /= relevantDocCount
		return avgPrecision

		# avgPrecision = -1

        # # Fill in code here
		# relPrecisions = []
		# if len(query_doc_IDs_ordered) == 0 or k < 1:
		# 	return -1
		# for i in range(min(k, len(query_doc_IDs_ordered))):
		# 	if str(query_doc_IDs_ordered[i]) in true_doc_IDs:
		# 		relPrecisions.append(self.queryPrecision(
		# 			query_doc_IDs_ordered, query_id, true_doc_IDs, i+1))

		# if len(relPrecisions) == 0:
		# 	print("No relevant documents")
		# 	avgPrecision = 0
		# else:
		# 	avgPrecision = sum(relPrecisions)/len(relPrecisions)

		# return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1
		if(len(doc_IDs_ordered)!=len(query_ids) or len(query_ids)==0):
			return meanAveragePrecision
		meanAveragePrecision = 0
		for i in range(len(query_ids)):
			meanAveragePrecision += self.queryAveragePrecision(doc_IDs_ordered[i], query_ids[i], q_rels, k)
		meanAveragePrecision /= len(query_ids)
		return meanAveragePrecision

