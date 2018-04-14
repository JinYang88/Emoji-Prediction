import pandas as pd

df = pd.read_csv("../data/tweet/multi/top20/tweet.csv")

top5 = sum(df['Label'].value_counts()[0:5])
top10 = sum(df['Label'].value_counts()[0:10])

print(top5/top10)

