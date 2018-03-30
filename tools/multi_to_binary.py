import pandas as pd


topn = 5
datasets = ['train','valid','test']
negtive = [0,1,2,3,4]
# datasets = ['test']

for dataset in datasets:
    data = pd.read_csv("../data/tweet/multi/top20/{}.csv".format(dataset))
    bi_data = []
    for idx, line in data.iterrows():
        if int(line['Label']) < topn:
            for i in negtive:
                bi_data.append([line['Id'], line['Text'], i, 1 if i == int(line['Label']) else -1])
            if int(line["Label"]) not in negtive:
                bi_data.append([line['Id'], line['Text'], int(line['Label']), 1])
    pd.DataFrame(bi_data, columns=["Id","Text","Emoji","Label"]).to_csv("../data/tweet/binary/top{}/{}.csv".format(topn, dataset), index=False)