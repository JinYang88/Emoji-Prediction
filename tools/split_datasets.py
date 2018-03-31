import pandas as pd


test_valid_percentage = 0.1
topn = 10

data = pd.read_csv("../data/tweet/multi/top20/tweet.csv")
data = data[data['Label'] < topn]
test_valid_lines = int(data.shape[0] * test_valid_percentage)

data = data[['Id','Text','Label']]
data.iloc[0:data.shape[0] - test_valid_lines * 2].to_csv("../data/tweet/multi/top{}/train.csv".format(topn), index=False)
data.iloc[data.shape[0] - test_valid_lines * 2 : data.shape[0] - test_valid_lines].to_csv("../data/tweet/multi/top{}/valid.csv".format(topn), index=False)
data.iloc[data.shape[0] - test_valid_lines: ].to_csv("../data/tweet/multi/top{}/test.csv".format(topn), index=False)
