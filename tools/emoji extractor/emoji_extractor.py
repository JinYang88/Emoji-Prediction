import pandas as pd
import re
import sys
import os
import json
import emojilib

def clean_text(text):
    #remove emojis, remove links, anonymize user mentions
    clean = ""
    text = emojilib.replace_emoji(text, replacement=' ')
    text = re.sub( '\s+', ' ', text).strip()
    for t in text.split(" "):
        if t.startswith('http'): 
            pass
        else:
            clean += t + " "
    #remove double spaces
    return clean

raw_data_path = "../../data/tweet/original/tweet_by_ID_20_3_2018__07_24_42.txt"
mapping = { '❤':'0' , '😍':'1' , '😂':'2' , '💕':'3' , '🔥':'4' , '😊':'5' , '😎':'6' , '✨':'7' , '💙':'8' , '😘':'9' , '📷':'10' , '🇺🇸':'11' , '☀':'12' , '💜':'13' , '😉':'14' , '💯':'15' , '😁':'16' , '🎄':'17' , '📸':'18' , '😜':'19'}
TopN = 20 # max:20

full_data = []
line_count = 0

with open(raw_data_path, 'r') as fr:
    for line in fr:
        line_count += 1
        js_dict = json.loads(line)
        
        Label = None
        data_line = None
        UserName = js_dict['user']['screen_name']
        Text = js_dict['text'].strip()
        Id = js_dict['id']
        emojis = emojilib.emoji_list(Text)
        Text = clean_text(Text)
        emoji_set = set([d['code'] for d in emojis if 'code' in d])
        if len(emoji_set) == 1:
            emoji = emoji_set.pop()
            if emoji in mapping:
                Label = mapping[emoji]
        if Label and int(Label) < TopN:
            data_line = [Id, UserName, Text, Label]
            full_data.append(data_line)
        if line_count % 10000 == 0:
            # print(data_line)
            print("Finish extracting {} lines.".format(line_count))
    full_data_df = pd.DataFrame(full_data, columns=["Id","UserName","Text","Label"])

full_data_df.to_csv("../../data/tweet/multi/top{}/tweet.csv".format(TopN), index=False)

print("Extracting from {} lines, {} lines are useful.".format(line_count, full_data_df.shape[0]))