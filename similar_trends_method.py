from gensim.models import KeyedVectors
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


model    = KeyedVectors.load('long_term_vectors_no_POS.kv')
model_st = KeyedVectors.load('short_term_vectors_no_POS.kv')



def mmr(doc_embedding: np.ndarray,
        word_embeddings: np.ndarray,
        words: List[str],
        top_n: int = 5,
        diversity: float = 0.8) -> List[str]:

    """
    Took this function from here -- (https://maartengr.github.io/BERTopic/api/mmr.html)

    Calculate Maximal Marginal Relevance (MMR) between candidate keywords and the document.

    MMR considers the similarity of keywords/keyphrases with the
    document, along with the similarity of already selected
    keywords and keyphrases. This results in a selection of keywords
    that maximize their within diversity with respect to the document.

    Parameters
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_n: The number of keywords/keyhprases to return
        diversity: How diverse the select keywords/keyphrases are.
                   Values between 0 and 1 with 0 being not diverse at all
                   and 1 being most diverse.

    Returns:
         List[str]: The selected keywords/keyphrases
    """

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]




def query_as_dict(model, parsed_qt: str, k_inner: int) -> Dict[str,float]:
    """
    Helper function to wrap up some of the logic for retrieving candidates, particularly
    error checking when a term is not recognized, and turning the list of results into a
    dictionary for joining.

    Parameters:
        The model being used to generate results
        A parsed query term
        Number of candidates to generate internally

    Returns dictionary mapping of candidates to their relevance scores
    """
    trends: List[ Tuple[str,float] ]
    try:
        trends = model.most_similar(parsed_qt, topn=k_inner)
    except KeyError:
        trends = []

    trends_to_scores: Dict[str,float] = defaultdict(int)
    for word, relevance in trends:
        trends_to_scores[word] = relevance

    return trends_to_scores



def safe_vec(model, key: str) ->  np.ndarray:
    """
    Parameters:
        the model to search in
        the word or phrase to search for in the two models

    returns a 100d numpy array. Contains 0s if the key is not present in the model
    """
    try:
        return model[key]
    except KeyError:
        return np.zeros(100)


def w2v_based_top_trends(qt: str, k: int = 20, k_inner: int = 200, L: float = 0.5):
    """
    Method outlined by the case interview. takes a query term as input, and searches
    short-term and long-term vector embeddings for results.

    Parameters:
        A query term
        Optionally the number of candidates to generate
        Optionnally the number of candidates to query internally, to be joined and abbreviated into the output list
        Optionally how much the returned list ought to prioritize diversity over similarity to the query term
    Returns an ordered list of related trends
    """
    assert k < k_inner
    parsed_qt = qt.lower().strip().replace(" ","_")

    # do a full join on these lists. i don't know a way to do in a fancy or less-verbose
    # way in python so
    st_scores = query_as_dict(model_st,parsed_qt,k_inner)
    lt_scores = query_as_dict(model, parsed_qt,k_inner)

    # do a left join on the two lists (just use scores from lt model. the intersection is typically quite small)
    all_words = set(lt_scores.keys())

    if len(all_words) == 0:
        return ['did not find anything']

    joined_list = [(word,st_scores[word],lt_scores[word]) for word in all_words]

    threshold = .90 # this filter is to remove synonyms as we are looking for similar
    # trends rather than exact word synonyms
    joined_filtered = [word for word,_,relevance in joined_list if relevance < threshold]


    # then sort by: (st * lt) / (st + lt)
    # but check first if the denominator is 0 to avoid a division error
    # safe_division = lambda tup: (tup[1]*tup[2])/(tup[1]+tup[2]) if (tup[1] + tup[2] != 0) else 0
    #joined_ordered = [w for w, st_score, lt_score in sorted(joined_list, key=safe_division, reverse=True)]

    ordered_diverse = mmr(
        doc_embedding   = [safe_vec(model, parsed_qt)],
        word_embeddings = [safe_vec(model, c) for c in joined_filtered],
        words = joined_filtered,
        top_n = k,
        diversity=L
    )
    return ordered_diverse
