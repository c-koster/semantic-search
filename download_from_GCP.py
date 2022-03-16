"""
The goal of this script is to randomly select 1_000_000 rows from the most recent
few months of data from the twitter and reddit datasets.

Try starting our select from November 10, 2020.
"""

import os

from google.cloud import bigquery
from google.oauth2 import service_account


path_base = os.getcwd()
key_path = path_base + '/nwo-sample-5f8915fdc5ec.json'
assert os.path.exists(key_path)

credentials = service_account.Credentials.from_service_account_file(
    key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
client = bigquery.Client(credentials=credentials, project=credentials.project_id,)

def download_query_data(QUERY: str, outfile: str) -> None:

    query_job = client.query(QUERY)
    rows = query_job.result()
    df = rows.to_dataframe(progress_bar_type='tqdm')
    df.to_parquet("{}/{}.parquet".format(path_base, outfile))
    print(df.head())



# set some consts
nrows = 1_000_000

# reddit timestamps are stored as integers.
unix_since_reddit: int = 1604966400
# created at is a string in the database. we can compare this way b/c later days are lexographically larger
time_since_twitter: str = '2020-11-10'
'''
from requests import get
import json
convert_url = "https://showcase.api.linx.twenty57.net/UnixTime/tounixtimestamp?datetime="
convert_str_unix_time = lambda ts: json.loads(get(convert_url + str(ts)).text) ['UnixTimeStamp']
'''


# first, get all my tweets read into a parquet
TWEETS_QUERY = """
SELECT tweet, created_at FROM `nwo-sample.graph.tweets`
WHERE created_at >= '{time_since}'
ORDER BY RAND()
LIMIT {nrows};
""".format(
    **{'time_since': time_since_twitter, 'nrows':nrows}
)
download_query_data(TWEETS_QUERY, "df_tweets")


# next, do the exact same thing with reddit
REDDIT_QUERY = """
SELECT body, created_utc FROM `nwo-sample.graph.reddit`
WHERE created_utc > {time_since}
ORDER BY RAND()
LIMIT {nrows};
""".format(
    **{'nrows':nrows,'time_since':unix_since_reddit}
)
download_query_data(REDDIT_QUERY, "df_reddit")
