import pandas as pd

test_valid_lines = 45000

data = pd.read_csv("../data/tweet/multi/top20/tweet.csv")
data = data[['Id','Text','Label']]
data.iloc[0:data.shape[0] - test_valid_lines * 2].to_csv("../data/tweet/multi/top20/train.csv", index=False)
data.iloc[data.shape[0] - test_valid_lines * 2 : data.shape[0] - test_valid_lines].to_csv("../data/tweet/multi/top20/valid.csv", index=False)
data.iloc[data.shape[0] - test_valid_lines: ].to_csv("../data/tweet/multi/top20/test.csv", index=False)
