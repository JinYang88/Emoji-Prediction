import pandas as pd
import os

postfixs = ['ids', 'labels', 'text']
dates = ['07_24']


Id = []
Text = []
Label = []

for dt in dates:
    for root,dirs,files in os.walk('../data/tweet/'):
        for file in files:
            filename = os.path.join(root, file)
            if dt in filename and not filename.endswith('txt'):
                print('Processing file [{}]'.format(filename))
                with open(filename, encoding='utf-8') as fr:
                    lines = [line.strip() for line in fr.readlines()]
                if 'ids' in filename:
                    Id.extend(lines)
                elif 'text' in filename:
                    Text.extend(lines)
                else:
                    Label.extend(lines)

out_csv = pd.DataFrame()
out_csv['Id'] = Id
out_csv['Text'] = Text
out_csv['Label'] = Label

out_csv.to_csv('../data/tweet/train.csv', index=False, encoding='utf-8')

