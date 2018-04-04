import pandas as pd
import os

topn = 5
datasets = ['train','valid','test']
negtive = [0,1,2,3,4]
all_recommend = range(topn)
# datasets = ['test']

if not os.path.isdir("../data/tweet/multi/top{}/".format(topn)):
    os.makedirs("../data/tweet/multi/top{}/".format(topn))

for dataset in datasets:
    data = pd.read_csv("../data/tweet/multi/top{}/{}.csv".format(topn, dataset))
    if dataset == "train":
        samples = negtive
    else:
        samples = all_recommend

    bi_data = []
    for idx, line in data.iterrows():
        if int(line['Label']) < topn:
            for i in samples:
                bi_data.append([line['Id'], line['Text'], i, 1 if int(line['Emoji']) == int(line['Label']) else -1])
            if int(line["Label"]) not in samples:
                bi_data.append([line['Id'], line['Text'], int(line['Label']), 1])
    pd.DataFrame(bi_data, columns=["Id","Text","Emoji","Label"]).to_csv("../data/tweet/multi/top{}/{}.csv".format(topn, dataset), index=False)