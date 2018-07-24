# -*- coding: utf-8 -*- 
import os
from pandas import Series, DataFrame
import numpy as np
import pickle

data_path = '/mnt/disk3/ohdonghun/Projects/CRC/data/chosun_utf8'

def NE_check(label_list):
    
    num_ne_list = [0] * len(label_list)
    for i, label in enumerate(label_list):
        num_ne = 0 
        for l in label:
            if l == 'B' or l == 'S':
                num_ne += 1
        num_ne_list[i] = num_ne
    
    return num_ne_list

def Len_check(label_list):
    len_list = [0] * len(label_list)
    for i, label in enumerate(label_list):
        len_list[i] = len(label)

    return len_list

# Read chn, label data
length = []
num_ne = []
chn_sent = []
label_sent = []
text_info = []

for (path, dir, files) in os.walk(data_path):
    for i, filename in enumerate(files): 
        ext = os.path.splitext(filename)[-1]
        if ext == '.chn':
            with open(os.path.join(path, filename), 'rb') as f:
                chn_sentences = pickle.load(f)
            with open(os.path.join(path, filename).replace('.chn','.label'), 'rb') as f:
                label_data_set = pickle.load(f)
            
            for _ in range(len(label_data_set)):
                text_info.append(filename.replace('.chn','.txt')) 
            length.extend(Len_check(label_data_set))
            num_ne.extend(NE_check(label_data_set))
            chn_sent.extend(chn_sentences)
            label_sent.extend(label_data_set)

print(len(length), len(num_ne), len(chn_sent), len(label_sent), len(text_info))

print(length[0:3])
print(num_ne[0:3])
print(chn_sent[0:3])
print(label_sent[0:3])
print(text_info[0:3])
base = {'Length' : length,
        'NE' : num_ne,
        'Hanmun' : chn_sent,
        'Label' : label_sent,
        'Text' : text_info}

data = DataFrame(base, columns=['Length','NE', 'Hanmun', 'Label', 'Text'])

data.to_csv('raw_NER_DB.csv', sep=',', encoding='utf-8', index=False)
