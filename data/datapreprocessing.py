import numpy as np
import os
import pickle
import pandas as pd
import codecs
import re

from sklearn.model_selection import train_test_split
from Named_Entity.utils.utils import check_path_exists, padding


def read_data(data_path):
    file_list = []
    for (path, dir, files) in os.walk(data_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.txt':
                file_list.append(os.path.join(path,filename))
    
    # Save the chosun file path list 
    with open('chosun_file_list.txt', 'wb') as f:
        pickle.dump(file_list, f)
    
    return file_list

def NE_detector(sentences):
    ne_dic = []
    for n, s in enumerate(sentences):
        
        s = s.replace(' ','').replace(';',':').replace('·','')
        length = 0
        start = None
        end = None
        for i, c in enumerate(s):
            if c == '(':
                start = i
            elif c == ')':
                end = i

            if start != None and end != None:
                check = s[start+1:end].find(':')
                if check != -1:
                    end = check
                length = end - start - 1
                
            if not length == 0:
                ko = s[start-length:start]
                ch = s[start+1:end]
                
                if ko.find('(') == -1 and ko.find(')') == -1 and ko.find('.') == -1:
                    ne_dic.append([ko,ch])
                 
                length = 0
                start = None
                end = None
    ne_dic = [list(ne) for ne in set(tuple(ne) for ne in ne_dic)]
   
    for i, ne in enumerate(ne_dic):
        if ne[0] == '':
            del ne_dic[i]
  
    return ne_dic

def combine_list(list_data):
    com_list = ''.join(list_data)

    return com_list

# make labels
def make_label(file_list):
    
    mark_total = ['!', '"', "'", '*', ',', '-', '.', '/', ':', ';', '?', '[', ']', '{', '}', '·', '‘', '’', '■', ' ',
          '▩','▲', '◎', '、', '〈', '〉', '《', '》', '「', '」', '『', '』', '【', '】', '〔', '〕', 'ㅿ', 'ㆍ', '：', '［', '］', '｢', '･'] 
    # read lines of each text
    for i,file in enumerate(file_list):
        if i % 1000 == 0: print('make labels {}/{}'.format(i,len(file_list)))
            
        with codecs.open(file,encoding='utf-8',mode='r') as f:
            lines = [] 
            for line in f.read().splitlines():
                if not line == '': 
                    
                    # remove mark                     
                    for m in mark_total: 
                        line = line.replace(m,'')
                     
                    lines.append(line)
      
         
        # search lines of text line by line 
        e = []
        for index, line in enumerate(lines):
            if line == '국역':
                k_start = index + 1
            elif line == '원문':
                k_end = index
                c_start = index + 1
            elif line.find('태백산사고본') != -1 or line.find('정족산사고본') != -1 \
                       or line.find('국편영인본') != -1 or line.find('원본') != -1:
                e.append(index)
        c_end = min(e)

        kor_lines = lines[k_start:k_end]
        ne_dic = NE_detector(kor_lines)
        chn_lines = lines[c_start:c_end]
        chn_sentences = chn_lines 
        ''' 
        chn_lines = combine_list(chn_lines)
        chn_sentences = chn_lines.split('。')
        ''' 
        label_list= []
        for c in range(len(chn_sentences)):
            
            #chn_sentences[c] +=  '。'    
                            

            # make a label 
            label = ['O'] * len(chn_sentences[c])
            for ne in ne_dic:
                if ne[0] != '':
                    find_check = chn_sentences[c].find(ne[1])
                     
                    if find_check != -1:   #find ne
                        # single cha NE 
                        if len(ne[1]) == 1:
                            label[find_check] = 'S'
                        else:
                            for n in range(len(ne[1])):
                                if n == 0:  # Begin
                                    label[find_check + n] = 'B'
                                elif n == len(ne[1])-1: # End
                                    label[find_check + n] = 'E'
                                else:   # Middle
                                    label[find_check + n] = 'I'
                               
            label = ''.join(label)  
            if len(chn_sentences[c]) != len(label):
                print('-----error-----------------------------')    
                print(file)

            label_list.append(label)

        # remove the empty list       
        chn_sentences = [chn_sent for chn_sent in chn_sentences if len(chn_sent) != 0]
        label_list = [label for label in label_list if len(label) != 0]
        

        if len(chn_sentences) != len(label_list):
            print('---------error------------')
            print(file)
        
        if i == 0:
            for chn, label in zip(chn_sentences, label_list):
                print(chn)
                print(label) 
        chn_text = file.replace('txt', 'chn') 
        label_text = file.replace('txt', 'label')
        ne_text = file.replace('txt', 'ne')
 
        # Save the chn sentence and label  
        with open(chn_text, 'wb') as f:
            pickle.dump(chn_sentences, f)
        
        with open(label_text, 'wb') as f:
            pickle.dump(label_list, f)
        with open(ne_text, 'wb') as f:
            pickle.dump(ne_dic, f)

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

def make_db(data_path):
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

    data = pd.DataFrame(base, columns=['Length','NE', 'Hanmun', 'Label', 'Text'])

    db_path = 'NER_DB.csv'
    data.to_csv(db_path, sep=',', encoding='utf-8', index=False)

    return db_path

def choice_db(data_path, max_len):
    
    data = pd.read_csv(data_path,sep=',', index_col=False)

    with_ne = data.loc[data['NE'] != 0]

    len_ne = with_ne.loc[data['Length'] <= max_len]

    db_path = 'NER_DB_choice.csv'
    len_ne.to_csv(db_path, sep=',', index=False)

    return db_path


def merge_data(data_path):
        
    # Read all of data & merge to one files
    # padding and split to train and test
    # Read a all of data and merge 
    print('Read csv & convert to list')
    data = pd.read_csv(data_path,sep=',', index_col=False)
    input_data_set = data['Hanmun'].tolist()
    label_data_set = data['Label'].tolist()
    length_data_set = data['Length'].tolist()
    print(length_data_set[0:3])

    for i,(chn, label) in enumerate(zip(input_data_set, label_data_set)):
        input_data_set[i] = list(chn)
        label_data_set[i] = list(label) 
    
    # padding
    print('Padding.....')
    input_data_set, label_data_set = padding(input_data_set, label_data_set, max_len=50)
    
    input_data_set = np.array(input_data_set)
    label_data_set = np.array(label_data_set) 

    print(input_data_set.shape)
    print(label_data_set.shape)

    print(input_data_set[0:3])

    print('pad total data : {}'.format(len(input_data_set)))
    print('pad total data : {}'.format(len(label_data_set)))
    
    train_inputs, test_inputs, train_label, test_label, train_len, test_len\
        = train_test_split(input_data_set, label_data_set, length_data_set, test_size = 0.3, random_state = 1)
    
    print(train_inputs.shape)
    print(test_inputs.shape)
    print(train_label.shape)
    print(test_label.shape)
    print(len(train_len))
    print(len(test_len))
 
    print('Save input and label data')
    
    train_path = 'train'
    test_path = 'test'
    
    check_path_exists([train_path, test_path])
                 
    # Save the input and label data

    with open(os.path.join(train_path,'chosun.chn'), 'wb') as f:
        pickle.dump(train_inputs, f)
    with open(os.path.join(train_path,'chosun.label'), 'wb') as f:
        pickle.dump(train_label, f)
    with open(os.path.join(train_path,'chosun.len'), 'wb') as f:
        pickle.dump(train_len, f)
    
    with open(os.path.join(test_path, 'chosun.chn'), 'wb') as f:
        pickle.dump(test_inputs, f)
    with open(os.path.join(test_path, 'chosun.label'), 'wb') as f:
        pickle.dump(test_label, f)
    with open(os.path.join(test_path, 'chosun.len'), 'wb') as f:
        pickle.dump(test_len, f)


# 1. read chosun & make label
data_path = '/mnt/disk3/ohdonghun/Projects/CRC/data/chosun_utf8'
file_list = read_data(data_path)
make_label(file_list)

# 2. make NER DB & choice condition(max_len, only with NE)
db_path = make_db(data_path)
db_path = choice_db(db_path, 50)    # max_len = 50

# 3. padding & split train, test data
merge_data(db_path)
