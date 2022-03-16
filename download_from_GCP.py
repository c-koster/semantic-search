import os

from google.cloud import bigquery
from google.oauth2 import service_account


path_base = os.getcwd()

key_path = path_base + '/nwo-sample-5f8915fdc5ec.json'

credentials = service_account.Credentials.from_service_account_file(
    key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
client = bigquery.Client(credentials=credentials, project=credentials.project_id,)


# first, get all my tweets read into a parquet
ALL_TWEETS = """
SELECT tweet, created_at FROM `nwo-sample.graph.tweets`
ORDER BY created_at DESC
LIMIT 100000;
"""
query_job = client.query(ALL_TWEETS)
rows = query_job.result()
df = rows.to_dataframe(progress_bar_type='tqdm')

print(df.head())

df.to_parquet(path_base + "/df_tweets.parquet")

# second do the exact same thing with reddit
ALL_REDDIT = """
SELECT body, created_utc FROM `nwo-sample.graph.reddit`
ORDER BY created_utc DESC
LIMIT 100000;
"""
query_job = client.query(ALL_REDDIT)
rows = query_job.result()
df = rows.to_dataframe(progress_bar_type='tqdm')

print(df.head())
df.to_parquet(path_base + "/df_reddit.parquet")
