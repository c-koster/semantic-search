from gensim.models import KeyedVectors
from collections import defaultdict


model    = KeyedVectors.load('long_term_vectors.kv')
model_st = KeyedVectors.load('short_term_vectors.kv')


def w2v_based_top_trends(qt: str, k: int = 20, k_inner: int = 200):
    # ensure the query term has been , and
    assert k < k_inner
    parsed_qt = qt.lower().strip().replace(" ","_")

    # do a full join on these lists. i don't know a way to do in a fancy or less-verbose way in python so here:
    try:
        long_term_trends  = model.most_similar(parsed_qt, topn=k)
        short_term_trends = model_st.most_similar(parsed_qt, topn=k)
    except KeyError:
        return [("did not find that trend",0)]

    st_scores = defaultdict(int)
    lt_scores = defaultdict(int)

    for word, relevance in short_term_trends:
        st_scores[word] = relevance

    for word, relevance in long_term_trends:
        lt_scores[word] = relevance


    # do a full join on these lists
    all_words = set([w for w,_ in long_term_trends] + [w for w,_ in short_term_trends])

    joined_list = [(word,st_scores[word],lt_scores[word]) for word in all_words]

    return [w for w, st_score, lt_score in sorted(joined_list, key=lambda tup: tup[2], reverse=True)[:k]]
