import sys
sys.path.append("../src/")
import datahelper
import gensim
import pandas as pd
import re

topn = 5

df = pd.read_csv('../data/tweet/multi/top5/train_with_emoji.csv')
sentences = []
valid_num = 0
for idx, line in df.iterrows():
    text = line['Text']
    try:
        number = int(re.findall(r"<\d+?>", text)[0].strip("<>"))
        valid_num += 1
    except:
        continue
    text = re.sub(r"(<\d>)+", " <{}> ".format(number), text)
    words = list(filter(lambda x: x!="", map(datahelper.normalizeString, text.split(' '))))
    sentences.append(words)
print("Use {} lines, {} lines valid".format(df.shape[0], valid_num))


model = gensim.models.Word2Vec(sentences, min_count=5, size=300,
window=3, iter=100, sg=0)
print(model)


with open("../data/embedding/top5embedding.txt", 'w') as fw:
    for k in model.wv.vocab:
        fw.write("{} {}\n".format(k, ' '.join(model[k].astype(str))))