"""
This script takes as input a reddit and twitter parquet file and outputs two trained
word2vec models (specifically gensim KeyedVector files). These are loaded in and
used by the similar trends method.

My process for creating these models is outlined briefly below (also see the readme)
1. load and preprocess:
    - tokenize and tag each row of text in my data using nltk's tweet tokenizer
    - keep the nouns and verbs, remove links and emojis
    - ( could paste the proper noun tags together here, but I opted to use a phraser later on)

2. concatenate the datasets and train a 'phraser model' to place an underscore between commonly occuring word groupings
    - do this on the text without hashtags and user tags, and append them later

3. train two w2v models, one short (past feew days of tweets) and one long term (past few months).
    - (later we will boost the ones that score highly on both)

"""

# imports
import os
from typing import List

# tagging and data storage
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag
from spacy.lang.en.stop_words import STOP_WORDS
import emoji

from gensim.models import Word2Vec


from tqdm import tqdm
tqdm.pandas() # .apply progress bars


# models
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec


# consts for tagging
STOP_EMOJI = set(emoji.UNICODE_EMOJI.keys())
STOP_WORDS_STOP_EMOJIS = STOP_WORDS.union(STOP_EMOJI)
GOOD_TAGS = {
    'NN','NNP','NNPS','NNS'#,'VBD','VB','VBG','VBN','VBP','VBZ'
}

path_base = os.getcwd()

# ensures the download_from_GCP.py file has been run already. i could import it too
assert os.path.exists(path_base + "/df_tweets.parquet")
assert os.path.exists(path_base + "/df_reddit.parquet")

# build dfs and make sure they are in the same time format
df_tweets = pd.read_parquet('/Users/cultonkoster/Desktop/homework/nwo-case-studies/nlp/df_tweets.parquet')
df_reddit = pd.read_parquet('/Users/cultonkoster/Desktop/homework/nwo-case-studies/nlp/df_reddit.parquet')
df_tweets["type"] = 1
df_reddit["type"] = 2


df_tweets['created_at'] = pd.to_datetime(df_tweets['created_at'])



to_int = lambda timedata: timedata.timestamp()
df_tweets['created_utc'] = df_tweets.pop('created_at').apply(to_int).astype('int64')


tweet_tokenizer = TweetTokenizer()

def preprocess_tweets(text: str) -> List[str]:
    """
    Pipeline for processing text. The trick here is to get all useful words into a
    sequence/list and then run gensim's phraser on the sentences. This is likely to be
    more useful as it does not rely on correct than pasting together proper NNP tags.

    Parameters:
        A string containing twitter text

    Returns a list of words.
    """
    tokens = tweet_tokenizer.tokenize(text)
    tags = pos_tag(tokens)

    words_out: List[str] = []

    for word,tag in tags:

        if tag not in GOOD_TAGS or word[:4] == "http" or len(word) < 3 or word in STOP_WORDS_STOP_EMOJIS:
            continue
        else:
            # we want to join contiguous proper nouns
            words_out.append(word.lower())

    return words_out


# run the above preprocessing function
print("pre-processing Twitter text")
df_tweets["text_preprocessed"] = df_tweets["tweet"].progress_apply(preprocess_tweets)

print("pre-processing Reddit text")
df_reddit["text_preprocessed"] = df_reddit["body"].progress_apply(preprocess_tweets)


# now add the two together before we extract phrases.. do this by calling their columns
# the same thing (text and created at) and stacking one on top of the other
df_all = pd.concat([df_tweets.drop('tweet',axis=1), df_reddit.drop('body',axis=1)])



# the last preprocessing thing I want to do before grouping:
# i've noticed in earlier iterations that hashtags and user tags (#,@) tend to get glued together,
# Let's remove these and then add them back in once phrasing is done.
hash_or_user_tag = lambda w: (w[0] == '#' or w[0] == '@')

filter_out  = lambda words: [word for word in words if not hash_or_user_tag(word)]
filter_only = lambda words: [word for word in words if hash_or_user_tag(word)]

df_all['words_no_@#'] = df_all['text_preprocessed'].apply(filter_out)
df_all['words_@#'] = df_all['text_preprocessed'].apply(filter_only)


phrases = Phrases(df_all['words_no_@#'], min_count=4, threshold=2)
phraser = Phraser(phrases)

def sentence_to_bi_grams(phrases_model, sentence: List[str]) -> List[str]:
    return phrases_model[sentence]

apply_phrases = lambda sentence: sentence_to_bi_grams(phraser,sentence)

df_all['text_with_phrases'] = df_all['words_no_@#'].apply(apply_phrases) + df_all['words_@#']

print("some examples of phrases for your viewing pleasure while the w2v models train")
for phrase, score in sorted(phrases.find_phrases(df_all['text_preprocessed']).items(), key=lambda tup: tup[1],reverse=True)[:20]:
    print(phrase, score)


model = Word2Vec(sentences=df_all["text_with_phrases"], vector_size=100, window=10, min_count=2, workers=4)
model.wv.save('long_term_vectors.kv')

# created_utc of 1610496000 is Wednesday, January 13, 2021 12:00:00 AM (GMT)
model_st = Word2Vec(sentences=df_all[df_all.created_utc > 1610496000]["text_with_phrases"])
model_st.wv.save('short_term_vectors.kv')
