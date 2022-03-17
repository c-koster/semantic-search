# Semantic Search Engine: NWO.ai Case Interview
1. [Introduction and Instructions](#introduction)
2. [Code Outline](#outline)
3. [Future Work and Considerations for More Data](#more)
    1. [Future Work](#future)
    2. [Other Approaches](#other)
    3. [Bigger Data](#bigdata)


## Introduction and Instructions to Run <a name="introduction"></a>
In this project, I implemented a semantic search engine for trend search, using a word2vec approach to retrieve a list of similar words/phrases to a user's query. This repo contains scripts for: loading data from GCP, preprocessing tweet and reddit text, and a method for finding trends which are similar to the queried trend. I've also included a test file to demonstrate a few searches.

Instructions on how to run and test models are below. This assumes that you have already placed a valid JSON key into the working directory named `nwo-sample-5f8915fdc5ec.json`.

```
pip3 install requirements.txt
python3 download_from_gcp.py
python3 make_gensim_models.py
python3 examples_trend_method.py
```

## Outline <a name="outline"></a>

1. `download_from_gcp.py`: contains a 
2. ``;
3. ``
4. `examples_trend_method.py`: this script imports the method from similar_trends_method.py and tries it out on a few trends.
5. `experiments_small.ipynb`: This is a notebook to show my experimenting work on a smaller version of the twitter and reddit dataset. 

## Future Work and Considerations for More Data<a name="more"></a>

### Sub paragraph <a name="future"></a>
1. Implement a separate pre-processing pipeline Reddit (Twitter has its own tokenizer but I didn't finnd one for reddit).
2. Find a bettter POS tagging tool for twwitter text.
3. Get a list of test queries: this way I could compare different appproches, or train a learning-to-rank model and re-rank the list provided by my method for better results.

### Sub paragraph <a name="other"></a>
I had two other ideas as a framework for extracting similar trends:

1. Try collaborative filtering, where tweets and Reddit comments are users, and items are individual words and phrases. 
2. Try out Associationn rule mining on a sparse matrix of words/phrases.

### Sub paragraph <a name="bigdata"></a>
The approach i've outlined here would certainly scale with more data: the gensim tools are designed to work on streams of text, so not everything needds to be held in memeory at once. However there are two exceptions. 

First, the data manipulation library that I used (pandas) holds all its data in memory, wwhich means that it will break if I add too many more rows. WWe would need to switch to a library which works on big data. I heard that [Spark](https://spark.apache.org/) and [Dask](https://dask.org/) have pandas APIs so this would take very little work to switch over.

Second, my preprocessing step is very slow (about 800it/s). This is because the nltk POS tagger takes a long time to label everything. This could be run in parallel or swapped out for a newer tool. 

