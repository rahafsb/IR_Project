import math
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from inverted_index_gcp import InvertedIndex
import numpy as np
import pandas as pd
from collections import Counter
import nltk
nltk.download('stopwords')

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["also", "category", "references", "external", "links", "first", "see", "history", "people", "one", "second", "may", "two", "part", "thumb", "including",  "following", "many", "however", "would", "became"]
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)
bucket_name = "209092196_212080188_209318666"


#tokenize the text while removing all stopwords and stemming each token id needed.
def tokenize(text, use_stemming=False):
  stemmer = PorterStemmer()
  Tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
  if use_stemming:
    Tokens = [stemmer.stem(token) for token in Tokens]
  return Tokens


def generate_query_tfidf(query, inverted_index: InvertedIndex):
  query_length = len(query)
  unique_tokens =np.unique(query)

  epsilon = .0000001
  query_vec = np.zeros(query_length)
  counter = Counter(query) #dictionary of each term eith its frequency
  for token in unique_tokens:
    if token in inverted_index.term_total.keys() and token in inverted_index.df.keys():  # avoid terms that do not appear in the index.
      tf = counter[token] / query_length  # total frequency per term divided by the length of the query
      df = inverted_index.df[token] # document frequency per term
      idf = math.log((len(inverted_index.doc_length)) / (df + epsilon), 10)  # smoothing
      try:
        ind = query.inverted_index(token)
        query_vec[ind] = tf * idf
      except:
        pass
  return query_vec



def cosine_similarity(doc_mat, query_vec):
    epsilon = .0000001
    dot_products = np.dot(doc_mat, query_vec)
    doc_norms = np.linalg.norm(doc_mat, axis=1)
    query_norm = np.linalg.norm(query_vec)
    cosine_similarities = {}
    for i, doc_id in enumerate(doc_mat.index):
      cosine_similarities[doc_id] = dot_products[i] / (doc_norms[i] * (query_norm+epsilon))

    return cosine_similarities


  # def generate_document_tfidf_matrix(query_to_search, index, comp):
def create_tfidf_matrix_for_query(query, search_index: InvertedIndex, compressor):
    # Identifying candidate documents and their scores for the query
    candidate_scores = find_documents_with_query_terms(query, search_index, compressor)
    # Extracting unique candidate document IDs
    unique_docs = np.unique([doc_id for doc_id, _ in candidate_scores.keys()])
    # Initializing the matrix with zeros
    tfidf_matrix = pd.DataFrame(np.zeros((len(unique_docs), len(query))), index=unique_docs, columns=query)
    # Populating the matrix with TF-IDF scores
    for (doc_id, term), tfidf_score in candidate_scores.items():
      tfidf_matrix.loc[doc_id, term] = tfidf_score
    return tfidf_matrix

  # get_candidate_documents_and_scores
def find_documents_with_query_terms(query,search_index: InvertedIndex, compressor):
    candidates = {}
    for token in query:
      if token in search_index.term_total:
        posting_list = search_index.read_a_posting_list(compressor, token, bucket_name)
        if posting_list:
          # Calculate normalized TF-IDF scores for documents containing the token
          normalized_tfidf = [
            (doc_id, (freq / search_index.doc_length[doc_id]) * math.log10(len(search_index.doc_length) / search_index.df[token])) for
            doc_id, freq in posting_list if freq > 0.1]
          # Populate the candidates dictionary with scores
          for doc_id, tfidf in normalized_tfidf:
            if tfidf > 0.1:
              candidates[(doc_id, token)] = candidates.get((doc_id, token), 0) + tfidf
    return candidates


def rank_documents_by_binary_similarity(search_index: InvertedIndex, query_tokens_, component_directory, n):
    query_tokens = tokenize(query_tokens_, True)
    scores = {}
    for token in query_tokens:
        posting_list = search_index.read_a_posting_list(component_directory,token, bucket_name)
        for doc_id, _ in posting_list:
            scores[doc_id] = scores.get(doc_id, 0) + 1 / len(query_tokens)

    ret = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ret[:n]

# nums = scores
def get_title(nums,titles):
    return [(x[0], titles[str(x[0])]) for x in nums]

def get_top_n_score_for_queries(queries_to_search, index: InvertedIndex, comp, n):
    queries_to_search=tokenize(queries_to_search,True)
    tfidf_matrix = create_tfidf_matrix_for_query(queries_to_search, index, comp)
    query_vec= generate_query_tfidf(queries_to_search,index)
    cosine_similarity_dict = cosine_similarity(tfidf_matrix,query_vec )
    sorted_similarities = sorted(cosine_similarity_dict.items(), key=lambda x: x[1], reverse=True)
    top_n = sorted_similarities[:n]
    return top_n






#######################################################

def merge_results(title_scores, body_scores, anchor_scores, pr,n=100, title_weight=0.45, body_weight=0.34,anchor_weight=0.04, pr_weight=0.12):
    """
    This function merge and sort documents retrieved by its weighted score (e.g., title and body).

    Parameters: ----------- title_scores: a dictionary build upon the title index of queries and tuples representing
    scores as follows: key: query_id value: list of pairs in the following format:(doc_id,score)

    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
    key: query_id value: list of pairs in the following format:(doc_id,score) title_weight: float, for weighted
    average utilizing title and body scores text_weight: float, for weighted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default, N = 3,
    for the topN function.

    Returns:
    -----------
    dictionary of queries and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score).
    """
    # Find the maximum score in the title, body, and anchor scores list

    if len(title_scores) != 0:
        max_score_title = max(title_scores, key=lambda x: x[1])[1]
    else:
        max_score_title = 1

    if len(body_scores) != 0:
        max_score_body = max(body_scores, key=lambda x: x[1])[1]
    else:
        max_score_body = 1

    if len(anchor_scores) != 0:
        max_score_anchor = max(anchor_scores, key=lambda x: x[1])[1]
    else:
        max_score_anchor = 1

    # Get all candidate doc_ids
    title_scores= dict(title_scores)
    body_scores= dict(body_scores)
    anchor_scores = dict(anchor_scores)
    all_candidate_docs = set(title_scores.keys()) | set(body_scores.keys()) | set(anchor_scores.keys())

    best_pr = []
    for wiki_id in all_candidate_docs:
        wiki_id = str(wiki_id)
        best_pr += [pr[wiki_id]] if wiki_id in pr else []

    max_pr = max(best_pr) if len(best_pr) != 0 else 1

    scores_merged_dict = {}
    # Loop through all the candidate documents
    for doc_id in all_candidate_docs:
        pr_score = pr[str(doc_id)] * pr_weight / max_pr
        # Calculate the scores for each metric

        try:
            if doc_id in title_scores:
                # Retrieve the score for the document ID
                title_score = title_scores[doc_id] * title_weight / max_score_title
            else:
                # Set the title score to 0 if the document ID is not found
                title_score = 0

            if doc_id in body_scores:
                # Retrieve the score for the document ID
                body_score = body_scores[doc_id] * body_weight / max_score_body
            else:
                # Set the title score to 0 if the document ID is not found
                body_score = 0

            if doc_id in anchor_scores:
                # Retrieve the score for the document ID
                anchor_score = anchor_scores[doc_id] * anchor_weight / max_score_anchor
            else:
                # Set the title score to 0 if the document ID is not found
                anchor_score = 0

            merged_score = title_score + body_score + anchor_score + pr_score
            scores_merged_dict[doc_id] = merged_score
        except KeyError:
            continue

    merged_scores = sorted([(doc_id, score) for doc_id, score in scores_merged_dict.items()], key=lambda x: x[1], reverse=True)[:n]

    return merged_scores