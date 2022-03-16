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
import os

path_base = os.getcwd()

# ensures the download_from_GCP.py file has been run already. i could import it too
assert os.path.exists(path_base + "/df_tweets.parquet")
assert os.path.exists(path_base + "/df_reddit.parquet")



from typing import List
# build dfs
