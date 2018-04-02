import pandas as pd
import re
import sys
import os
import json
sys.path.append('./emoji extractor')
import emojilib

def clean_text(text, replacement):
    #remove emojis, remove links, anonymize user mentions
    clean = ""
    text = emojilib.replace_emoji(text, replacement=replacement)
    text = re.sub( '\s+', ' ', text).strip()
    for t in text.split(" "):
        if t.startswith('http'): 
            pass
        else:
            clean += t + " "
    #remove double spaces
    return clean


mapping = { '❤':'0' , '😍':'1' , '😂':'2' , '💕':'3' , '🔥':'4' , '😊':'5' , '😎':'6' , '✨':'7' , '💙':'8' , '😘':'9' , '📷':'10' , '🇺🇸':'11' , '☀':'12' , '💜':'13' , '😉':'14' , '💯':'15' , '😁':'16' , '🎄':'17' , '📸':'18' , '😜':'19'}


topn = 5

ori_df = pd.read_csv("../data/tweet/multi/top{}/train.csv".format(topn))
ori_ids = set(ori_df['Id'])


raw_data_path = "../data/tweet/original/tweet_by_ID_20_3_2018__07_24_42.txt"


list_with_emoji = []
linecount = 0
with open(raw_data_path, 'r') as fr:
    for line in fr:
        linecount = +=1
        js_dict = json.loads(line)
        
        Label = None
        data_line = None
        UserName = js_dict['user']['screen_name']
        Text = js_dict['text'].strip()
        Id = js_dict['id']
        if Id in ori_ids:
            emojis = emojilib.emoji_list(Text)
            emoji_set = set([d['code'] for d in emojis if 'code' in d])
            emoji = emoji_set.pop()
            Text = clean_text(Text, "<{}>".format(mapping[emoji]))
            list_with_emoji.append([Id, Text])
        if linecount % 10000:
            print("Processed {} lines.".format(linecount))

df_with_emoji = pd.DataFrame(list_with_emoji)
df_with_emoji.to_csv("../data/tweet/multi/top{}/train_with_emoji.csv".format(topn), header=['Id','Text'] ,index=False)