"""




"""
import os

path_base = os.getcwd()

# ensures the download_from_GCP.py file has been run already. i could import it too
assert os.path.exists(path_base + "/df_tweets.parquet")
assert os.path.exists(path_base + "/df_reddit.parquet")



# build dfs
