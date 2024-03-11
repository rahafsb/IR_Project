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
    """
    Tokenizes the input text by removing all stopwords and optionally stemming each token.

    Parameters:
    - text (str): The text to tokenize.
    - use_stemming (bool, optional): If True, applies stemming to tokens. Defaults to False.

    Returns:
    - list: A list of processed tokens from the input text.
    """
    stemmer = PorterStemmer()
    Tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    if use_stemming:
        Tokens = [stemmer.stem(token) for token in Tokens]
    return Tokens


def cosine_similarity(doc_mat, query_vec):
    """
      Calculates the cosine similarity between a matrix of document vectors and a query vector.

      Parameters:
      - doc_mat (numpy.ndarray): A matrix where each row represents a document vector.
      - query_vec (numpy.ndarray): A vector representing the query.

      Returns:
      - dict: A dictionary mapping document identifiers to their cosine similarity scores with the query.
      """
    epsilon = .0000001
    dot_products = np.dot(doc_mat, query_vec)
    doc_norms = np.linalg.norm(doc_mat, axis=1)
    query_norm = np.linalg.norm(query_vec)
    cosine_similarities = {}
    for i, doc_id in enumerate(doc_mat.index):
      cosine_similarities[doc_id] = dot_products[i] / (doc_norms[i] * (query_norm+epsilon))

    return cosine_similarities


def create_tfidf_matrix_for_query(query, search_index: InvertedIndex, compressor):
    """
        function constructs a TF-IDF (Term Frequency-Inverse Document Frequency) matrix for a specified query,
        using a given search index and a compressor to identify and score candidate documents.

        Parameters:
        -query: List of query terms.
        -search_index (InvertedIndex): The search index used to locate documents containing the query terms.
        -compressor: Used in the process of finding documents, though its exact role is not specified in the snippet.

        Returns:
        -tfidf_matrix (pd.DataFrame): A pandas DataFrame where rows represent unique documents
        containing at least one query term, columns represent query terms, and values are the TF-IDF scores indicating
         the relevance of each term in each document to the query.

          """
    candidate_scores = find_documents_with_query_terms(query, search_index, compressor)
    unique_docs = np.unique([doc_id for doc_id, _ in candidate_scores.keys()])
    tfidf_matrix = pd.DataFrame(np.zeros((len(unique_docs), len(query))), index=unique_docs, columns=query)
    for (doc_id, term), tfidf_score in candidate_scores.items():
      tfidf_matrix.loc[doc_id, term] = tfidf_score
    return tfidf_matrix


def generate_query_tfidf(query, inverted_index: InvertedIndex):
    """
    Generates a TF-IDF vector for a query based on an inverted index.

    Parameters:
    - query (list of str): The search query as a list of tokens.
    - inverted_index (InvertedIndex): An instance of InvertedIndex containing indexed documents.

    Returns:
    - numpy.ndarray: A vector representing the TF-IDF scores for the query.
    """
    eps = 0.00000001
    query_length = len(query)
    unique_tokens = np.unique(query)
    query_vec = np.zeros(query_length)
    counter = Counter(query)
    for token in unique_tokens:
        if token in inverted_index.term_total.keys() and token in inverted_index.df.keys():
            tf = counter[token] / query_length
            df = inverted_index.df[token]
            idf = math.log((len(inverted_index.doc_length)) / (df + eps), 10)
            ind = query.inverted_index(token)
            query_vec[ind] = tf * idf
    return query_vec



def rank_documents_by_binary_similarity(search_index: InvertedIndex, query_tokens_, component_directory, n):
    """
        function ranks documents based on their binary similarity to a set of query tokens, using a given search index

        Parameters:
        -search_index (InvertedIndex): The search index used to locate documents.
        -query_tokens_: The raw query input, which is a string or list of strings that needs to be tokenized.
        -component_directory: The directory (or specific component) used with the search index to read posting lists.
        -n: The number of top documents to return based on their similarity scores.

        Returns:
        -ret (list): A list of tuples where each tuple contains a document
        ID and its similarity score, sorted in descending order
         of similarity. Only the top n documents are returned.
        """

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
    """
     function retrieves titles for a given set of document IDs from a titles mapping.

     Parameters:
     -nums: A list of tuples, where each tuple contains a document ID and its similarity score.
     -titles (dict): A dictionary mapping document IDs (as strings) to their respective titles.

     Returns:
     A list of tuples, where each tuple contains a document
     ID and its corresponding title, for the document IDs provided in nums.
     """
    return [(x[0], titles[str(x[0])]) for x in nums]

def get_top_n_score_for_queries(queries_to_search, index: InvertedIndex, comp, n):
    """
    explan:
      Finds and ranks the top 'n' documents based on their cosine similarity to a given set of query terms.
      It tokenizes and optionally stems the query terms, creates a TF-IDF matrix for these terms using the given search index,
      generates a query TF-IDF vector, calculates cosine similarities between this vector and document vectors in the matrix,
      and sorts the documents based on these similarities to return the top 'n' documents.

    Parameters:
      - queries_to_search: The query or set of queries to be searched.
      - index (InvertedIndex): The search index used for finding and scoring documents.
      - comp: The compressor used for creating the TF-IDF matrix. Its exact role is not specified, but it's involved in the TF-IDF calculation or retrieval process.
      - n (int): The number of top-scoring documents to return.

    Returns:
      - A list of tuples, each containing a document ID and its cosine similarity score, representing the top 'n' documents sorted by their relevance to the query.
    """
    queries_to_search=tokenize(queries_to_search,True)
    tfidf_matrix = create_tfidf_matrix_for_query(queries_to_search, index, comp)
    query_vec= generate_query_tfidf(queries_to_search,index)
    cosine_similarity_dict = cosine_similarity(tfidf_matrix,query_vec )
    sorted_similarities = sorted(cosine_similarity_dict.items(), key=lambda x: x[1], reverse=True)
    top_n = sorted_similarities[:n]
    return top_n




#def merge_results(title_scores, body_scores, anchor_scores, pr,n=100, title_weight=0.40, body_weight=0.36,anchor_weight=0.08, pr_weight=0.16):
# Find the maximum score in the title, body, and anchor scores list
def merge_results(title_scores):
    """
       explan:
         Merges and normalizes scores from title matches, producing a ranked list of document IDs based on their relevance.
         This function takes pre-calculated title scores, normalizes these scores using the highest title score and a predefined weight,
         and returns the top 'n' documents sorted by these normalized scores. The approach highlights the importance of titles in determining document relevance.

       Parameters:
         - title_scores: A list of tuples or a dictionary, where each tuple or key-value pair represents a document ID and its associated score from title matching.

       Returns:
         - A list of tuples, each containing a document ID and its normalized score, sorted by score in descending order. The list includes the top 'n' ranked documents based on title relevance.
       """

    n = 100
    title_weight = 0.99
    if len(title_scores) != 0:
        max_score_title = max(title_scores, key=lambda x: x[1])[1]
    else:
        max_score_title = 1

    title_scores= dict(title_scores)

    all_candidate_docs = set(title_scores.keys())
    # best_pr = []
    # for wiki_id in all_candidate_docs:
    #     wiki_id = str(wiki_id)
    #     best_pr += [pr[wiki_id]] if wiki_id in pr else []
    #
    # max_pr = max(best_pr) if len(best_pr) != 0 else 1

    scores_merged_dict = {}
    # Loop through all the candidate documents
    for doc_id in all_candidate_docs:
       # pr_score = pr[str(doc_id)] * pr_weight / max_pr
        # Calculate the scores for each metric

        try:
            if doc_id in title_scores:
                # Retrieve the score for the document ID
                title_score = title_scores[doc_id] * title_weight / max_score_title
            else:
                # Set the title score to 0 if the document ID is not found
                title_score = 0

            # if doc_id in body_scores:
            #     # Retrieve the score for the document ID
            #     body_score = body_scores[doc_id] * body_weight / max_score_body
            # else:
            #     # Set the title score to 0 if the document ID is not found
            #     body_score = 0
            #
            # if doc_id in anchor_scores:
            #     # Retrieve the score for the document ID
            #     anchor_score = anchor_scores[doc_id] * anchor_weight / max_score_anchor
            # else:
            #     # Set the title score to 0 if the document ID is not found
            #     anchor_score = 0

            merged_score = title_score #+ body_score + anchor_score + pr_score
            scores_merged_dict[doc_id] = merged_score
        except KeyError:
            continue

    merged_scores = sorted([(str(doc_id), score) for doc_id, score in scores_merged_dict.items()], key=lambda x: x[1], reverse=True)[:n]

    return merged_scores


def find_documents_with_query_terms(query,search_index: InvertedIndex, compressor):
    """
        function identifies and scores documents based on their relevance to given query terms,
        using a specified search index and compressor.

        Parameters:
        -query: List of query terms.
        -search_index (InvertedIndex): The search index used for locating documents that contain the query terms.
        -compressor: Utilized for reading posting lists from the search index.

        Returns:
        -candidates (dict): A dictionary where keys are tuples of document ID and query term, and values are the corresponding
        TF-IDF scores indicating the relevance of each document to the query terms.
        """
    candidates = {}
    for token in query:
      if token in search_index.term_total.keys():
        posting_list = search_index.read_a_posting_list(compressor, token, bucket_name)
        if posting_list:
          normalized_tfidf = [
            (doc_id, (freq / search_index.doc_length[doc_id]) * math.log10(len(search_index.doc_length) / search_index.df[token])) for
            doc_id, freq in posting_list if freq > 0.1]
          for doc_id, tfidf in normalized_tfidf:
            if tfidf > 0.1:
              candidates[(doc_id, token)] = candidates.get((doc_id, token), 0) + tfidf
    return candidates
