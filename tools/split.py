import pandas as pd

data = pd.read_csv("../data/tweet/multi/top20/tweet.csv")
data = data[['Id','Text','Label']]
data.iloc[0:data.shape[0]-10000].to_csv("../data/tweet/multi/top20/train.csv", index=False)
data.iloc[data.shape[0]-10000:data.shape[0]-5000].to_csv("../data/tweet/multi/top20/valid.csv", index=False)
data.iloc[data.shape[0]-5000:].to_csv("test.csv", index=False)
