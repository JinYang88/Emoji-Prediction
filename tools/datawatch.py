import pandas as pd

df = pd.read_csv("../data/tweet/multi/top20/tweet.csv")
print(df['Label'].value_counts())