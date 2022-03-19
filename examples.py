from similar_trends_method import w2v_based_top_trends


K = 20
test_terms = ["augmented_reality", "iphone", "elon musk", "bike", "joe biden",
"kombucha", "bitcoin", "student loans","gold","pepsi","virtual reality","@realdonaldtrump",
"canada_goose","cognac","cbd", "punk ipa", "#denial", "sandbox"
]


data = []
for i in test_terms:
    trends = w2v_based_top_trends(i,K)
    print( "Query Term: {}\nResults: {}\n".format(i, trends) )
    data.append([i] + trends)


import pandas as pd

df = pd.DataFrame.from_records(
    data,
    columns=['queryterm'] + ["rank_{}".format(ranknum) for ranknum in range(1,K+1)]
)

df.to_csv('example_queries.csv',index=False)
