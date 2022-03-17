from similar_trends_method import w2v_based_top_trends


K = 20
test_terms = ["augmented_reality", "iphone", "elon musk", "bike", "joe biden",
"kombucha", "bitcoin", "student loans","gold","pepsi","virtual reality","@realdonaldtrump"]

for i in test_terms:
    trends = w2v_based_top_trends(i,K)
    print( "Query Term: {}\nResults: {}\n".format(i, trends) )
