import torch
import re
import torch


class BatchWrapper:
    def __init__(self, dl, iter_columns):
        self.dl, self.iter_columns = dl, iter_columns # we pass in the list of attributes for x &amp;amp;amp;amp;lt;g class="gr_ gr_3178 gr-alert gr_spell gr_inline_cards gr_disable_anim_appear ContextualSpelling ins-del" id="3178" data-gr-id="3178"&amp;amp;amp;amp;gt;and y&amp;amp;amp;amp;lt;/g&amp;amp;amp;amp;gt;

    def __iter__(self):
        for batch in self.dl:
            yield (getattr(batch, attr) for attr in self.iter_columns)
  
    def __len__(self):
        return len(self.dl)

def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"<br />",r" ",s)
    s = re.sub(r'(\W)(?=\1)', '', s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"@.*$", r" ", s)

    return s


def load_glove_as_dict(filepath):
    word_vec = {}
    with open(filepath) as fr:
        for line in fr:
            line = line.split()
            word = line[0]
            vec = line[1:]
            word_vec[word] = vec
    return word_vec