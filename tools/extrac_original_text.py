import pandas as pd

topn = 5
filetype = "binary"

ori_df = pd.read_csv("../data/tweet/{}/top{}/train.csv".format(filetype, topn))
full_df = pd.read_csv("../data/tweet/multi/top20/tweet.csv")


ori_ids = ori_df['Id']
new_df = full_df[full_df['Id'].isin(ori_ids)]
new_df.to_csv(index=False)