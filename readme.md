# Semantic Search Engine: NWO.ai Case Interview
1. [Introduction and Instructions](#introduction)
2. [Code Outline](#outline)
3. [Future Work and Considerations for More Data](#more)
    1. [Future Work](#future)
    2. [Other Approaches](#other)
    3. [Bigger Data](#bigdata)


## Introduction and Instructions to Run <a name="introduction"></a>
In this project, I implemented a semantic search engine for trend search, using a word2vec approach to retrieve a list of similar words/phrases to an input query. I trained a 'long-term' and short-term model, which used data across different time windows. (However in practice the results list is sorted by the long-term model only, as the short-term vectors anecdotally produce nonsensical results.)

This repo contains scripts for: loading data from GCP, preprocessing tweet and Reddit text, and a method for finding trends which are similar to the queried trend. I've also included a test file to demonstrate a few searches.

Instructions on how to run and test models are below. This assumes that you have already placed a valid JSON key into the working directory named `nwo-sample-5f8915fdc5ec.json`.

```
pip3 install -r requirements.txt
python3 download_from_gcp.py
python3 make_gensim_models.py
python3 examples.py
```

## Outline <a name="outline"></a>

- `download_from_gcp.py`: this script randomly selects a sample of 1 million recent rows from Twitter and Reddit databases on google cloud. Then it writes them to a parquet file for preprocessing, phrasing, and training. 
- `make_gensim_models.py`: this script preprocesses and trains the two word2vec models used in for semantic search method, and then saves copies of the trained models in the working directory. The long-term model is trained on the whole dataset (a few months), while the short-term model creates embeddings for data found in the past few weeks. Specific preprocessing steps are listed below:
    1. tokenizing data using a Twitter-specific tokenizer from NLTK
    2. data is run through an NLTK POS tagger
    3. non-stopword nouns are kept and passed into a phrasing model

- `similar_trends_method.py`: this script implements the method requested in the case interview description——it works by taking 'short term' and long term word2vec models, taking the corresponding vector from the query term, and finding the most similar wodd/phrase vectors in both models. Then, the method joins short and long term lists and returns the top K results.
- `examples.py`: this script imports the method from similar_trends_method.py and tries it out on a few trends.
- ~`experiments_small.ipynb`: this is a notebook to show my experimenting work on a smaller version of the twitter and reddit dataset.~

## Future Work and Considerations for More Data<a name="more"></a>

### Future Work <a name="future"></a>
1. Implement a separate pre-processing pipeline Reddit (Twitter has its own tokenizer but I didn't find one for Reddit).
2. Find a bettter (and faster) POS tagging tool for Twitter text.
3. Get a list of 'test' queries: this way I could compare different appproches, or train a learning-to-rank model and re-rank the list provided by my method for better results.

### Other Approaches <a name="other"></a>
I had two other ideas of frameworks to try out for extracting similar trends:

1. Try collaborative filtering, where tweets and Reddit comments are users, and items are individual words and phrases. 
2. Try out Association rule mining on a sparse matrix of words/phrases.

### Bigger Data <a name="bigdata"></a>
The approach I've outlined here would certainly scale to a larger text corpus: the gensim tools that I used are designed to work on streams of text, so not everything needds to be held in memory at once. However there are two exceptions. 

First, the data manipulation library that I used (pandas) holds all its data in memory, so it might break if I add too many more rows. I would need to switch to a library which holds data data on disk and streams into memory when needed. I've heard that [Spark](https://spark.apache.org/) and [Dask](https://dask.org/) have pandas APIs so it would take little work to switch over.

Second, my preprocessing step is very slow (about 800it/s). which is because the NLTK POS tagger takes a long time to label each word. Solutions couldd involve running this task in parallel, or swapping out for a quicker POS tagger.
